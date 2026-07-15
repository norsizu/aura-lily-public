/**
 * 消息协议辅助
 */
#include "messages.h"
#include <string.h>

static const char *POSE_NAMES[] = {
    "idle_a", "idle_b", "listen_a",
    "listen_b", "thinking", "speak_a",
    "speak_b", "happy", "proud",
};

static const char *SCENE_NAMES[] = {
    "living_room", "bedroom", "study",
};

int msg_pose_to_index(const char *name)
{
    for (int i = 0; i < 9; i++) {
        if (strcmp(name, POSE_NAMES[i]) == 0) return i;
    }
    /* 未知名称返回 -1：不覆盖设备本地随机姿势 */
    return -1;
}

int msg_scene_to_index(const char *name)
{
    for (int i = 0; i < 3; i++) {
        if (strcmp(name, SCENE_NAMES[i]) == 0) return i;
    }
    return 0;
}

const char *msg_index_to_pose(int index)
{
    if (index >= 0 && index < 9) return POSE_NAMES[index];
    return "idle_a";
}
