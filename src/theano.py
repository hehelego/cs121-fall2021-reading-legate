import numpy
import aesara
import aesara.tensor as aet
rng = numpy.random
N, feats, training_steps = 400, 784, 10000

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# Declare Aesara symbolic variables
x, y = aet.dmatrix("x"), aet.dvector("y")
# shared variables are kept between training iterations
# initialize the weight vector w randomly
w = aesara.shared(rng.randn(feats), name="w")
# initialize the bias term
b = aesara.shared(0., name="b")

# Construct Aesara expression graph
# Probability that target = 1
p_1 = 1 / (1 + aet.exp(-T.dot(x, w) - b))
# The prediction thresholded
prediction = p_1 > 0.5
# Cross-entropy loss function
xent = -y * aet.log(p_1) - (1-y) * aet.log(1-p_1)
# The cost to minimize
cost = xent.mean() + 0.01 * (w ** 2).sum()
# Compute the gradient of the cost w.r.t weight vector w and bias term b
gw, gb = aet.grad(cost, [w, b])

# Compile
train = aesara.function(inputs=[x, y], outputs=[prediction, xent],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = aesara.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
