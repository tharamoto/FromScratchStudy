import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)



class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported.'.format(type(data)))
            

        self.data       = data
        self.name       = name
        self.grad       = None
        self.creator    = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        
        p = str(self.data).replace('¥n', '¥n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator    = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
            
        #funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            #gys = [output.grad for output in f.outputs] # 出力の微分をリストにする
            gys = [output().grad for output in f.outputs] # 出力の微分をリストにする
            gxs = f.backward(*gys) # アンパックして渡す
            if not isinstance(gxs, tuple): # タブルでない場合はタプルにする
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs): # 入力と出力の微分を取得
                if x.grad is None:
                    x.grad = gx # 入力変数に微分を設定
                else:
                    x.grad = x.grad + gx # 入力変数に微分を加算

                if x.creator is not None:
                    #funcs.append(x.creator) # さらに上流の関数も取得、ここでの関数追加でスタックが積みあがる
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs: # 出力の微分を削除
                    y().grad = None # yは弱参照なので()をつけて実体を取り出す
                     
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs] # 入力がリストの場合
        ys = self.forward(*xs) # アンパックして渡す
        if not isinstance(ys, tuple): # タプルでない場合はタプルにする
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys] # 出力もリストの場合

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in  outputs: # 出力がリストの場合
                output.set_creator(self) # 出力変数に生みの親を覚えさせる

            self.inputs = inputs # 入力も覚える
            #self.outputs = outputs # 出力も覚える
            self.outputs = [weakref.ref(output) for output in outputs] # 出力を弱参照で覚える

        return outputs if len(outputs) > 1 else outputs[0] # 出力が1つだけならVariableを返す

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)



class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy 

    
def add(x0, x1):
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy * x1
        gx1 = gy * x0
        return gx0, gx1
    

def mul(x0, x1):
    return Mul()(x0, x1)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = add(mul(a, b), c)

y.backward()

print(y)
print(a.grad)
print(b.grad)




