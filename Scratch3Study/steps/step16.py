import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported.'.format(type(data)))
            

        self.data       = data
        self.grad       = None
        self.creator    = None
        self.generation = 0

    def set_creator(self, func):
        self.creator    = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
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
            gys = [output.grad for output in f.outputs] # 出力の微分をリストにする
            gxs = f.backward(*gys) # アンパックして渡す
            if not isinstance(gxs, tuple): # タブルでない場合はタプルにする
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs): # 入力と出力の微分を取得
                if x.grad is None:
                    x.grad = gx # 入力変数に微分を設定
                else:
                    x.grad = x.grad + gx # 入力変数に微分を加算

                if x.creator is not None:
                    #funcs.append(x.creator) # さらに上流の関数も取得
                    add_func(x.creator)

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

        self.generation = max([x.generation for x in inputs])
        for output in  outputs: # 出力がリストの場合
            output.set_creator(self) # 出力変数に生みの親を覚えさせる

        self.inputs = inputs # 入力も覚える
        self.outputs = outputs # 出力も覚える

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


x = Variable(np.array(2.0))
y = add(x, x)
y.backward()
print(x.grad)

#x = Variable(np.array(3.0))
x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)