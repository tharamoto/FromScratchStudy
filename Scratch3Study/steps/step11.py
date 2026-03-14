import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported.'.format(type(data)))
            

        self.data       = data
        self.grad       = None
        self.creator    = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 関数を取得
            x, y = f.input, f.output # 関数の入力と出力を取得
            x.grad = f.backward(y.grad) # 関数のbackwardメソッド
            if x.creator is not None:
                funcs.append(x.creator) # さらに上流の関数も取得

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs] # 入力がリストの場合
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys] # 出力もリストの場合
        
        for output in outputs: # 出力がリストの場合
            output.set_creator(self) # 出力変数に生みの親を覚えさせる

        self.inputs = inputs # 入力も覚える
        self.outputs = outputs # 出力も覚える

        return outputs
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)