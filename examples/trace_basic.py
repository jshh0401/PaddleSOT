import numpy as np

import paddle
from sot.translate import symbolic_translate


def foo(x: paddle.Tensor, y: paddle.Tensor):
    z = x + y
    return z + 1


def main():
    x = paddle.rand([2, 3])
    y = paddle.rand([2, 3])
    # 动态图输出
    dygraph_out = foo(x, y)
    # 动转静态图输出, 调用symbolic_translate, 传入foo函数, 自动转写
    # -> sot/translate.py:21
    symbolic_translate_out = symbolic_translate(foo)(x, y)

    print("dygraph_out:", dygraph_out)
    print("symbolic_translate_out:", symbolic_translate_out)
    np.testing.assert_allclose(
        dygraph_out.numpy(), symbolic_translate_out.numpy()
    )


if __name__ == '__main__':
    main()
