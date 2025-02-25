from __future__ import annotations

import dis
import sys
from functools import partial

from ..profiler import EventGuard
from ..utils import CodeStatus, log, log_do
from .custom_code import CustomCode
from .executor.executor_cache import OpcodeExecutorCache
from .skip_files import need_skip


def print_locals(frame):
    local_key = [
        key for key in frame.f_locals.keys() if not key.startswith("__")
    ]
    print(
        f"[eval_frame_callback] {frame.f_code.co_name} with locals {local_key}"
    )
    print(
        f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} with cellvars + freevars:  {frame.f_code.co_cellvars + frame.f_code.co_freevars}"
    )

    def convert_obj(obj):
        import paddle

        if isinstance(obj, paddle.Tensor):
            return "Tensor(" + str(obj.shape) + ")"
        if isinstance(obj, list):
            return [convert_obj(i) for i in obj]
        return obj

    for key in local_key:
        print(
            f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} {key} = {convert_obj(frame.f_locals[key])}"
        )


def eval_frame_callback(frame, **kwargs) -> CustomCode:
    # -> sot/profiler.py:26 EventGuard定义
    # 记录栈帧相关信息
    with EventGuard(
        f"eval_frame_callback: {frame.f_code.co_name}", event_level=2
    ):
        # is generator
        if frame.f_code.co_flags & 0x20 > 0:
            return CustomCode(None, True)

        # NOTE(SigureMo): Temporary fallback when code has exception handling.
        if sys.version_info >= (3, 11) and frame.f_code.co_exceptiontable:
            log(
                3,
                f"[eval_frame_callback] {frame.f_code} has co_exceptiontable\n",
            )
            return CustomCode(None, False)

        if need_skip(frame):
            log(3, f"[eval_frame_callback] skip {frame.f_code}\n")
            custom_code = CustomCode(None, False)
            new_code = frame.f_code
        else:
            log(
                2, f"[eval_frame_callback] start to translate: {frame.f_code}\n"
            )
            log_do(4, partial(print_locals, frame))

            log(3, f"[transform] OriginCode: {frame.f_code.co_name}\n")
            log_do(3, lambda: dis.dis(frame.f_code))
            
            # 核心翻译部分
            custom_code = OpcodeExecutorCache()(frame, **kwargs)

            if custom_code.code is None:
                log(
                    3,
                    "[transform] NewCode (same as origin code): "
                    + frame.f_code.co_name
                    + "\n",
                )
                new_code = frame.f_code
            else:
                log(
                    3,
                    "[transform] NewCode: " + custom_code.code.co_name + "\n",
                )
                log_do(3, lambda: dis.dis(custom_code.code))
                new_code = custom_code.code

        # just check those codes which need open eval_frame
        if (
            custom_code.disable_eval_frame is False
            and CodeStatus().is_code_without_graph(new_code)
        ):
            log(
                3,
                "[eval_frame_callback] Code has no graph, block it.\n",
            )
            return CustomCode(None, True)

        return custom_code
