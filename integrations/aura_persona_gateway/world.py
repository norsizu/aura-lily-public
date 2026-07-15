from __future__ import annotations

import datetime as dt
import hashlib
import random
import time
from typing import Any

from .time_context import resolve_city_tzinfo


WORLD_SCHEMA_VERSION = "lily_world_v1"
WORLD_DISABLED_REASON = "world_model_disabled"
ORDINARY_REPLY_REASON = "ordinary_reply"
# metadata 里缓存的 generated current 只在这个窗口内可信，过期后回落到
# 当天计划重新推导（否则"吃早饭"会一直挂到晚上）。
GENERATED_CURRENT_TTL_SECONDS = 30 * 60.0


def build_world_snapshot(
    *,
    config: Any,
    store: Any,
    state: dict[str, Any],
    query_context: dict[str, Any],
    user_geo: dict[str, Any] | None = None,
    voice_low_latency: bool = False,
    recent_messages: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    enabled = bool(getattr(config, "world_model_enabled", True))
    city = _clean_text(getattr(config, "aura_home_city", ""))
    ts = float(now if now is not None else time.time())
    # day_key 按 Lily 所在城市的当地时间计算，与容器时区无关。
    day_key = dt.datetime.fromtimestamp(ts, resolve_city_tzinfo(city)).date().isoformat()
    if voice_low_latency and not _query_needs_world_context(query_context):
        current = _manual_current_from_state(state) or {
            "location_key": "home",
            "location_label": "家里",
            "activity_key": "idle",
            "activity_label": "安静待着",
            "status": "compact_voice_fallback",
            "city": city,
            "source": "compact_voice_fallback",
            "available": True,
        }
        policy = _mention_policy(
            query_context=query_context,
            state=state,
            current=current,
            recent_messages=recent_messages or [],
            disabled=not enabled,
        )
        compact = {
            "enabled": enabled,
            "available": enabled,
            "city": city,
            "day_key": day_key,
            "current": current if enabled else {},
            "today_plan": [],
            "next_plan": None,
            "mention_policy": policy,
            "debug": {},
        }
        return {
            **compact,
            "prompt_lines": render_world_prompt(compact).splitlines() if enabled else [],
            "debug": {
                "schema": WORLD_SCHEMA_VERSION,
                "reason": "compact_voice_no_world_query" if enabled else WORLD_DISABLED_REASON,
                "build_ms": _elapsed_ms(started),
                "generated_new_plan": False,
                "plan_count": 0,
                "current_source": current.get("source") or "",
                "voice_low_latency": True,
                "user_geo_source": str((user_geo or {}).get("source") or ""),
            },
        }
    if not enabled:
        return {
            "enabled": False,
            "available": False,
            "city": city,
            "day_key": day_key,
            "current": {},
            "today_plan": [],
            "next_plan": None,
            "mention_policy": _mention_policy(
                query_context=query_context,
                state=state,
                current={},
                recent_messages=recent_messages or [],
                disabled=True,
            ),
            "prompt_lines": [],
            "debug": {
                "reason": WORLD_DISABLED_REASON,
                "build_ms": _elapsed_ms(started),
                "voice_low_latency": bool(voice_low_latency),
                "user_geo_source": str((user_geo or {}).get("source") or ""),
            },
        }

    scope = config.scope
    plan = store.today_plan(scope, day_key=day_key)
    generated_new_plan = False
    if not _is_lily_world_plan(plan):
        plan = generate_day_plan(
            city=city,
            day_key=day_key,
            state=state,
            now=ts,
        )
        if hasattr(store, "replace_day_plan"):
            store.replace_day_plan(scope, day_key=day_key, items=plan)
        generated_new_plan = True
    plan = _with_status(plan, now=ts)
    if hasattr(store, "save_day_plan_statuses"):
        store.save_day_plan_statuses(scope, day_key=day_key, items=plan)

    current = _manual_current_from_state(state)
    current_source = "manual_state" if current else ""
    if not current:
        current = _generated_current_from_state(state, city=city, now=ts)
        current_source = "generated_state" if current else "generated_plan"
    if not current:
        current = _current_from_plan(plan, now=ts, city=city)
        _save_generated_current(config, store, state, current, now=ts)
    next_plan = _next_plan(plan, now=ts)
    policy = _mention_policy(
        query_context=query_context,
        state=state,
        current=current,
        recent_messages=recent_messages or [],
        disabled=False,
    )
    prompt_lines = render_world_prompt(
        {
            "enabled": True,
            "available": True,
            "city": city,
            "day_key": day_key,
            "current": current,
            "today_plan": plan,
            "next_plan": next_plan,
            "mention_policy": policy,
            "debug": {},
        }
    ).splitlines()
    return {
        "enabled": True,
        "available": True,
        "city": city,
        "day_key": day_key,
        "current": current,
        "today_plan": plan,
        "next_plan": next_plan,
        "mention_policy": policy,
        "prompt_lines": prompt_lines,
        "debug": {
            "schema": WORLD_SCHEMA_VERSION,
            "build_ms": _elapsed_ms(started),
            "generated_new_plan": generated_new_plan,
            "plan_count": len(plan),
            "current_source": current_source,
            "voice_low_latency": bool(voice_low_latency),
            "user_geo_source": str((user_geo or {}).get("source") or ""),
        },
    }


def generate_day_plan(
    *,
    city: str,
    day_key: str,
    state: dict[str, Any] | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    seed = int(hashlib.sha256(f"{WORLD_SCHEMA_VERSION}:{city}:{day_key}".encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    energy = _coerce_int((state or {}).get("energy"), 70)
    mood = _coerce_int((state or {}).get("mood"), 80)
    satiety = _coerce_int((state or {}).get("satiety"), 80)
    base_date = dt.date.fromisoformat(day_key)
    outing_axis = _choose_outing_axis(rng, energy=energy, mood=mood)
    lunch_out = energy >= 48 and satiety <= 88 and rng.random() > 0.35
    afternoon_out = outing_axis["location_key"] != "home"
    slots = [
        _slot(
            base_date,
            slot_key="wake",
            minutes=7 * 60 + 35 + rng.randint(-25, 35),
            duration=45,
            activity_type="morning",
            title=_pick(rng, ["醒来整理一下", "慢慢起床", "洗漱换好衣服"]),
            location_key="home",
            location_label="家里",
            activity_label="刚起床",
            life_axis="home",
            city=city,
        ),
        _slot(
            base_date,
            slot_key="breakfast",
            minutes=8 * 60 + 20 + rng.randint(-20, 25),
            duration=35,
            activity_type="meal",
            title=_pick(rng, ["吃点早饭", "简单吃早饭", "把早饭解决掉"]),
            location_key="home",
            location_label="家里",
            activity_label="吃早饭",
            life_axis="meal",
            city=city,
        ),
        _slot(
            base_date,
            slot_key="morning_focus",
            minutes=10 * 60 + rng.randint(-35, 30),
            duration=105 + rng.randint(-15, 25),
            activity_type="quiet",
            title=_pick(rng, ["上午安静处理点事情", "上午在屋里待一会儿", "把上午留给安静的事"]),
            location_key="desk",
            location_label=_pick(rng, ["书桌边", "窗边", "房间里"]),
            activity_label=_pick(rng, ["整理东西", "看点内容", "安静待着"]),
            life_axis="quiet",
            city=city,
        ),
        _slot(
            base_date,
            slot_key="lunch",
            minutes=12 * 60 + 15 + rng.randint(-25, 45),
            duration=55,
            activity_type="meal",
            title=_pick(rng, ["吃午饭", "午饭时间", "找点东西吃"]),
            location_key="neighborhood_food" if lunch_out else "home",
            location_label=_pick(rng, ["附近小店", "住处附近"]) if lunch_out else "家里",
            activity_label="吃午饭",
            life_axis="meal",
            city=city,
        ),
        _slot(
            base_date,
            slot_key="afternoon",
            minutes=15 * 60 + rng.randint(-30, 60),
            duration=80 + rng.randint(-10, 35),
            activity_type=outing_axis["activity_type"],
            title=outing_axis["title"],
            location_key=outing_axis["location_key"],
            location_label=outing_axis["location_label"],
            activity_label=outing_axis["activity_label"],
            life_axis=outing_axis["life_axis"],
            city=city,
        ),
        _slot(
            base_date,
            slot_key="evening",
            minutes=18 * 60 + 10 + rng.randint(-20, 45),
            duration=75,
            activity_type="home" if afternoon_out else "quiet",
            title=_pick(rng, ["回到家里缓一会儿", "傍晚回家休息"]) if afternoon_out else _pick(rng, ["傍晚在家放松", "傍晚收拾一下"]),
            location_key="home",
            location_label="家里",
            activity_label="休息",
            life_axis="home",
            city=city,
            extra={"requires_prior_outing": afternoon_out},
        ),
        _slot(
            base_date,
            slot_key="dinner",
            minutes=19 * 60 + 10 + rng.randint(-25, 35),
            duration=45,
            activity_type="meal",
            title=_pick(rng, ["吃晚饭", "晚饭", "晚上吃点东西"]),
            location_key="home",
            location_label="家里",
            activity_label="吃晚饭",
            life_axis="meal",
            city=city,
        ),
        _slot(
            base_date,
            slot_key="night_settle",
            minutes=22 * 60 + 35 + rng.randint(-30, 45),
            duration=80,
            activity_type="rest",
            title=_pick(rng, ["睡前整理一天", "晚上慢慢收尾", "准备休息"]),
            location_key="home",
            location_label="家里",
            activity_label="睡前整理",
            life_axis="rest",
            city=city,
        ),
    ]
    slots.sort(key=lambda item: float(item.get("scheduled_at") or 0))
    return _with_status(slots, now=float(now if now is not None else time.time()))


def render_world_prompt(snapshot: dict[str, Any] | None) -> str:
    if not snapshot or not snapshot.get("enabled"):
        return ""
    policy = snapshot.get("mention_policy") if isinstance(snapshot.get("mention_policy"), dict) else {}
    current = snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {}
    plan = snapshot.get("today_plan") if isinstance(snapshot.get("today_plan"), list) else []
    next_plan = snapshot.get("next_plan") if isinstance(snapshot.get("next_plan"), dict) else None
    allow_location = bool(policy.get("allow_location"))
    allow_activity = bool(policy.get("allow_activity"))
    allow_plan = bool(policy.get("allow_plan"))
    location_precision = str(policy.get("location_precision") or "none")
    lines = [
        f"世界模型=启用；Lily 所在城市={snapshot.get('city') or '未设置'}。",
        "本轮世界信息可见策略="
        + f"location:{allow_location} activity:{allow_activity} plan:{allow_plan}；"
        + f"precision={location_precision}；"
        + f"reason={policy.get('reason') or ORDINARY_REPLY_REASON}。",
    ]
    if allow_location or allow_activity:
        current_text = _current_text(current, expose_location=allow_location, expose_activity=allow_activity)
        if current_text:
            lines.append("当前状态：" + current_text)
        if allow_activity and not allow_location:
            lines.append("本轮只允许透露当前活动或模糊状态；不要补充具体地点、在家、商场、店铺、路线或食物。")
    else:
        lines.append("当前地点、活动和日计划只作为内部背景；本轮不要主动说出具体地点、店铺、商场、食物或行程。")
    if allow_plan:
        plan_lines = [_plan_line(item, expose_location=allow_location) for item in plan[:10]]
        if plan_lines:
            lines.append("今日计划：\n" + "\n".join(f"- {item}" for item in plan_lines if item))
    elif next_plan and policy.get("reason") != "relationship_boundary":
        lines.append("如果用户追问今天安排，可以优先用当前/下一项计划；未被问到时不要主动展开。")
    if policy.get("reason") == "relationship_boundary":
        lines.append("用户在问位置/行程，但当前关系边界不适合透露具体位置；可以自然、简短、模糊地回答，不要像客服式拒绝。")
    if policy.get("intent") in {"weather", "weather_advice"} and policy.get("subject_entity") == "aura":
        lines.append("天气回答必须优先使用本地天气缓存；没有 fresh 缓存时承认暂时没有实时天气，不要编造。")
    return "\n".join(line for line in lines if line).strip()


def world_debug_event(snapshot: dict[str, Any]) -> dict[str, Any]:
    policy = snapshot.get("mention_policy") if isinstance(snapshot.get("mention_policy"), dict) else {}
    return {
        "schema": WORLD_SCHEMA_VERSION,
        "enabled": bool(snapshot.get("enabled")),
        "available": bool(snapshot.get("available")),
        "city": snapshot.get("city"),
        "day_key": snapshot.get("day_key"),
        "current": snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {},
        "next_plan": snapshot.get("next_plan") if isinstance(snapshot.get("next_plan"), dict) else None,
        "mention_policy": policy,
        "debug": snapshot.get("debug") if isinstance(snapshot.get("debug"), dict) else {},
    }


def _choose_outing_axis(rng: random.Random, *, energy: int, mood: int) -> dict[str, str]:
    if energy < 34:
        return {
            "activity_type": "rest",
            "title": _pick(rng, ["下午在家休息", "下午不出门，缓一缓"]),
            "location_key": "home",
            "location_label": "家里",
            "activity_label": "休息",
            "life_axis": "rest",
        }
    options = [
        {
            "activity_type": "walk",
            "title": _pick(rng, ["出门走一小圈", "去附近透透气"]),
            "location_key": "nearby_walk",
            "location_label": _pick(rng, ["住处附近", "附近小公园", "街边"]),
            "activity_label": "散步",
            "life_axis": "breath",
        },
        {
            "activity_type": "quiet",
            "title": _pick(rng, ["找个安静地方待会儿", "下午去附近坐一会儿"]),
            "location_key": "quiet_stop",
            "location_label": _pick(rng, ["附近咖啡店", "安静角落"]),
            "activity_label": "安静停留",
            "life_axis": "quiet",
        },
        {
            "activity_type": "errand",
            "title": _pick(rng, ["顺路买点日用品", "去附近补点东西"]),
            "location_key": "daily_shop",
            "location_label": _pick(rng, ["附近便利店", "社区小店"]),
            "activity_label": "日常购物",
            "life_axis": "daily_shopping",
        },
    ]
    if mood >= 78 and energy >= 58:
        options.append(
            {
                "activity_type": "browse",
                "title": _pick(rng, ["去附近商场逛一会儿", "下午随便逛逛"]),
                "location_key": "mall",
                "location_label": "附近商场",
                "activity_label": "随便逛逛",
                "life_axis": "browse",
            }
        )
    return dict(rng.choice(options))


def _slot(
    base_date: dt.date,
    *,
    slot_key: str,
    minutes: int,
    duration: int,
    activity_type: str,
    title: str,
    location_key: str,
    location_label: str,
    activity_label: str,
    life_axis: str,
    city: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    minutes = max(0, min(23 * 60 + 55, int(minutes)))
    # 槽位时间是"城市当地时间"，转 epoch 必须带城市时区，不能吃容器时区。
    start = dt.datetime.combine(
        base_date,
        dt.time(hour=minutes // 60, minute=minutes % 60),
        tzinfo=resolve_city_tzinfo(city),
    )
    payload = {
        "world_schema": WORLD_SCHEMA_VERSION,
        "duration_minutes": max(15, int(duration)),
        "location_key": location_key,
        "location_label": location_label,
        "activity_label": activity_label,
        "life_axis": life_axis,
        "city": city,
        "source": "lily_world_model",
    }
    if extra:
        payload.update(extra)
    return {
        "plan_date": base_date.isoformat(),
        "slot_key": slot_key,
        "scheduled_at": start.timestamp(),
        "activity_type": activity_type,
        "title": title,
        "location": location_label,
        "should_post": 0,
        "status": "pending",
        "expected_delta": {},
        "payload": payload,
    }


def _with_status(plan: list[dict[str, Any]], *, now: float) -> list[dict[str, Any]]:
    out = []
    for item in sorted(plan, key=lambda row: float(row.get("scheduled_at") or 0)):
        row = dict(item)
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        duration = max(15, _coerce_int(payload.get("duration_minutes"), 45))
        start = float(row.get("scheduled_at") or 0)
        end = start + duration * 60
        if now < start:
            status = "pending"
        elif now <= end:
            status = "active"
        else:
            status = "done"
        row["status"] = status
        if status == "done" and not row.get("executed_at"):
            row["executed_at"] = end
        out.append(row)
    return out


def _current_from_plan(plan: list[dict[str, Any]], *, now: float, city: str) -> dict[str, Any]:
    if not plan:
        return {
            "location_key": "home",
            "location_label": "家里",
            "activity_key": "idle",
            "activity_label": "安静待着",
            "status": "fallback",
            "city": city,
            "source": "fallback_home",
            "available": True,
        }
    active = [item for item in plan if item.get("status") == "active"]
    if active:
        return _current_from_slot(active[0], city=city, source="active_plan")
    past = [item for item in plan if float(item.get("scheduled_at") or 0) <= now]
    if past:
        return _current_from_slot(past[-1], city=city, source="nearest_plan")
    return _current_from_slot(plan[0], city=city, source="next_plan")


def _current_from_slot(slot: dict[str, Any], *, city: str, source: str) -> dict[str, Any]:
    payload = slot.get("payload") if isinstance(slot.get("payload"), dict) else {}
    return {
        "location_key": _clean_text(payload.get("location_key") or slot.get("activity_type") or "home"),
        "location_label": _clean_text(payload.get("location_label") or slot.get("location") or "家里"),
        "activity_key": _clean_text(slot.get("activity_type") or "idle"),
        "activity_label": _clean_text(payload.get("activity_label") or slot.get("title") or "安静待着"),
        "title": _clean_text(slot.get("title") or ""),
        "slot_key": _clean_text(slot.get("slot_key") or ""),
        "status": _clean_text(slot.get("status") or ""),
        "city": city,
        "source": source,
        "available": True,
    }


def _next_plan(plan: list[dict[str, Any]], *, now: float) -> dict[str, Any] | None:
    for item in sorted(plan, key=lambda row: float(row.get("scheduled_at") or 0)):
        if float(item.get("scheduled_at") or 0) > now:
            return dict(item)
    return None


def _query_needs_world_context(query_context: dict[str, Any]) -> bool:
    intent = str((query_context or {}).get("intent") or "").strip()
    subject = str((query_context or {}).get("subject_entity") or "").strip()
    boundary = str((query_context or {}).get("boundary") or "").strip()
    if intent in {"whereabouts", "day_plan", "weather", "weather_advice", "time", "time_weather"}:
        return True
    if subject == "aura" and boundary:
        return True
    return False


def _manual_current_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    if str(metadata.get("world_current_source") or "") != "manual" and not metadata.get("world_manual_override"):
        return None
    activity = _clean_text(metadata.get("current_activity") or "")
    location_key = _clean_text(metadata.get("current_location") or state.get("scene") or "")
    location_label = _clean_text(metadata.get("location_label") or location_key)
    if not (activity or location_label):
        return None
    return {
        "location_key": location_key or "manual",
        "location_label": location_label or location_key or "手动位置",
        "activity_key": "manual",
        "activity_label": activity or "手动状态",
        "title": activity or "手动状态",
        "slot_key": "manual_override",
        "status": "manual",
        "city": "",
        "source": "manual_state",
        "available": True,
    }


def _generated_current_from_state(state: dict[str, Any], *, city: str, now: float) -> dict[str, Any] | None:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    if str(metadata.get("world_current_source") or "") != "generated":
        return None
    # 过期检查：缓存的 current 只是当时计划槽位的快照，随时间必须失效，
    # 否则会一直卡在旧活动上（时区修复前生成的"吃早饭"就是这样残留的）。
    try:
        age = float(now) - float(metadata.get("world_last_updated_at"))
    except (TypeError, ValueError):
        return None
    if age < 0 or age > GENERATED_CURRENT_TTL_SECONDS:
        return None
    activity = _clean_text(metadata.get("current_activity") or "")
    location_key = _clean_text(metadata.get("current_location") or state.get("scene") or "")
    location_label = _clean_text(metadata.get("location_label") or location_key)
    if not (activity or location_label):
        return None
    return {
        "location_key": location_key or "generated",
        "location_label": location_label or location_key or "生成位置",
        "activity_key": "generated",
        "activity_label": activity or "生成状态",
        "title": activity or "生成状态",
        "slot_key": "generated_state",
        "status": "generated",
        "city": city,
        "source": "generated_state",
        "available": True,
    }


def _save_generated_current(config: Any, store: Any, state: dict[str, Any], current: dict[str, Any], *, now: float) -> None:
    if not current:
        return
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    if str(metadata.get("world_current_source") or "") == "manual" or metadata.get("world_manual_override"):
        return
    updated = dict(state)
    next_metadata = dict(metadata)
    next_metadata["current_activity"] = _clean_text(current.get("activity_label") or "")
    next_metadata["current_location"] = _clean_text(current.get("location_key") or "")
    next_metadata["location_label"] = _clean_text(current.get("location_label") or "")
    next_metadata["world_current_source"] = "generated"
    next_metadata["world_last_updated_at"] = float(now)
    next_metadata["world_schema"] = WORLD_SCHEMA_VERSION
    updated["metadata"] = next_metadata
    store.save_state(config.scope, updated)


def _mention_policy(
    *,
    query_context: dict[str, Any],
    state: dict[str, Any],
    current: dict[str, Any],
    recent_messages: list[dict[str, Any]],
    disabled: bool,
) -> dict[str, Any]:
    intent = str(query_context.get("intent") or "chat")
    subject = str(query_context.get("subject_entity") or "unknown")
    query_boundary = str(query_context.get("boundary") or "")
    if disabled:
        return {
            "allow_location": False,
            "allow_activity": False,
            "allow_moment": False,
            "allow_plan": False,
            "location_precision": "none",
            "reason": WORLD_DISABLED_REASON,
            "intent": intent,
            "subject_entity": subject,
            "boundary": query_boundary,
        }
    asks_current = intent == "activity_or_location" and subject == "aura"
    asks_plan = intent == "day_plan" and subject == "aura"
    asks_aura_weather_or_time = intent in {"weather", "weather_advice", "time", "time_weather"} and subject == "aura"
    relationship_gate = _relationship_boundary(state)
    if asks_aura_weather_or_time:
        allow_location = False
        allow_activity = False
        allow_plan = False
        location_precision = "city"
        reason = "asked_aura_weather_or_time"
    elif asks_current or asks_plan:
        allow_location = relationship_gate["allow_specific_location"]
        allow_activity = relationship_gate["allow_activity"]
        allow_plan = asks_plan and relationship_gate["allow_plan"]
        location_precision = "specific" if allow_location else "vague"
        reason = "asked_world" if asks_current else "asked_day_plan"
        if not (allow_location or allow_activity or allow_plan):
            reason = "relationship_boundary"
    else:
        allow_location = False
        allow_activity = False
        allow_plan = False
        location_precision = "none"
        reason = ORDINARY_REPLY_REASON
    if not (asks_current or asks_plan) and _recently_mentioned(current, recent_messages):
        allow_location = False
        allow_activity = False
        allow_plan = False
        location_precision = "none"
        reason = "recent_location_cooldown"
    return {
        "allow_location": allow_location,
        "allow_activity": allow_activity,
        "allow_moment": False,
        "allow_plan": allow_plan,
        "location_precision": location_precision,
        "reason": reason,
        "intent": intent,
        "subject_entity": subject,
        "boundary": query_boundary,
        "relationship_gate": relationship_gate,
    }


def _relationship_boundary(state: dict[str, Any]) -> dict[str, Any]:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    flags = metadata.get("relationship_flags") if isinstance(metadata.get("relationship_flags"), dict) else {}
    trust = _coerce_int(state.get("trust"), 50)
    affinity_xp = _coerce_int(state.get("affinity_xp"), 0)
    social_need = _coerce_int(metadata.get("social_need"), 0)
    privacy = _coerce_int(metadata.get("privacy_sensitivity"), 35)
    strained = bool(flags.get("strained"))
    closeness = trust + min(70, affinity_xp // 4) + min(30, social_need // 2) - privacy
    allow_plan = not strained and closeness >= 40
    allow_activity = not strained and closeness >= 48
    allow_specific_location = not strained and closeness >= 70 and trust >= 58 and affinity_xp >= 80 and privacy <= 72
    return {
        "trust": trust,
        "affinity_xp": affinity_xp,
        "social_need": social_need,
        "privacy_sensitivity": privacy,
        "strained": strained,
        "closeness": closeness,
        "allow_plan": allow_plan,
        "allow_activity": allow_activity,
        "allow_specific_location": allow_specific_location,
    }


def _recently_mentioned(current: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    location = _clean_text(current.get("location_label") or "")
    if not location or len(location) < 2:
        return False
    recent_aura = [
        str(item.get("body") or "")
        for item in messages[-6:]
        if str(item.get("direction") or "") == "aura"
    ]
    return any(location in body for body in recent_aura)


def _is_lily_world_plan(plan: list[dict[str, Any]]) -> bool:
    if not plan:
        return False
    lily_rows = 0
    for item in plan:
        title = _clean_text(item.get("title") or "")
        if "陪伴时光" in title:
            return False
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if payload.get("world_schema") == WORLD_SCHEMA_VERSION:
            lily_rows += 1
    return lily_rows >= max(1, len(plan) // 2)


def _current_text(current: dict[str, Any], *, expose_location: bool, expose_activity: bool) -> str:
    parts = []
    if expose_location and current.get("location_label"):
        parts.append(f"位置={current.get('location_label')}")
    if expose_activity and current.get("activity_label"):
        parts.append(f"活动={current.get('activity_label')}")
    if current.get("status"):
        parts.append(f"状态={current.get('status')}")
    return " ".join(str(item) for item in parts if item)


def _plan_line(item: dict[str, Any], *, expose_location: bool) -> str:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    when = _fmt_time(item.get("scheduled_at"), city=str(payload.get("city") or ""))
    title = _clean_text(item.get("title") or "")
    location = _clean_text(item.get("location") or "") if expose_location else ""
    status = _clean_text(item.get("status") or "")
    return " ".join(part for part in (when, status, title, f"@{location}" if location else "") if part)


def _fmt_time(value: Any, *, city: str = "") -> str:
    try:
        return dt.datetime.fromtimestamp(float(value), resolve_city_tzinfo(city)).strftime("%H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def _pick(rng: random.Random, items: list[str]) -> str:
    return str(rng.choice(items))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
