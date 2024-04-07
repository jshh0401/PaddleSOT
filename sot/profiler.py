import os
from contextlib import contextmanager
from functools import wraps

from paddle.framework import core

_event_level = int(os.environ.get("EVENT_LEVEL", "-1"))


class SotProfiler:
    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def enable(self, tag=None):
        core.nvprof_start()
        core.nvprof_enable_record_event()

    def disable(self):
        core.nvprof_stop()


@contextmanager
def EventGuard(event_name, event_level=0):
    try:
        # 全局的事件等级
        global _event_level
        # 标记是否存在需要出栈的事件
        need_pop = False
        # 目前还不知道event_level具体代表啥
        if _event_level >= event_level:
            # 这里对应c++代码, 在pybind暴露
            # [](const std::string &name) platform::CudaNvtxRangePush(name, platform::NvtxRangeColor::Green)
            """
            void CudaNvtxRangePush(const std::string& name, const NvtxRangeColor color) {
                nvtxEventAttributes_t eventAttrib;
                eventAttrib.version = NVTX_VERSION;
                eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                eventAttrib.colorType = NVTX_COLOR_ARGB;
                eventAttrib.color = static_cast<uint32_t>(color);
                eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
                eventAttrib.message.ascii = name.c_str();

                dynload::nvtxRangePushEx(&eventAttrib);
            }
            """
            # 看起来像入栈了一些事件有关的信息
            core.nvprof_nvtx_push(event_name)
            need_pop = True
        yield
    finally:
        if need_pop:
            core.nvprof_nvtx_pop()


if _event_level == -1:

    @contextmanager
    def _EmptyEventGuard(event_name, event_level=0):
        yield

    EventGuard = _EmptyEventGuard  # noqa: F811


def event_register(event_name, event_level=0):
    def event_wrapper(func):
        @wraps(func)
        def call_with_event(*args, **kwargs):
            with EventGuard(event_name, event_level=0):
                return func(*args, **kwargs)

        return call_with_event

    def do_nothing(func):
        return func

    global _event_level
    if _event_level >= event_level:
        return event_wrapper
    else:
        return do_nothing
