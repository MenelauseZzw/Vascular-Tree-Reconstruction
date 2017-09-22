import h5py
import numpy as np
import numpy.linalg as linalg
from scipy.sparse.linalg import LinearOperator,lsqr
import tensorflow as tf

class ProblemBuilder:
    def __init__(self, measurements, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, lambd):
        self.numDimensions = measurements.shape[1]
        self.measurements = measurements
        self.tangentLinesPoints1 = tangentLinesPoints1
        self.tangentLinesPoints2 = tangentLinesPoints2
        self.numPoints = len(measurements)
        self.radiuses = radiuses
        self.indices1 = indices1
        self.indices2 = indices2
        self.lambd = lambd

    def getVariablePlaceholder(self, dtype):
        numVars = self.Parametrization.numParams(self.numDimensions) * self.numPoints
        return tf.placeholder(dtype, (numVars,))

    def getVariableValue(self):
        return np.vstack((self.tangentLinesPoints1, self.tangentLinesPoints2)).ravel()

    def getFunction(self, x):
        measurements = self.measurements

        indices1 = self.indices1
        indices2 = self.indices2

        s,t = tf.split(x, num_or_size_splits=2)
        s = tf.reshape(s, [-1, self.numDimensions])
        t = tf.reshape(t, [-1, self.numDimensions])

        distSq = self.Parametrization(s, t).euclideanDistanceSq(measurements)
        dist = tf.sqrt(tf.clip_by_value(distSq, clip_value_min=0.001, clip_value_max=1.0)) / self.radiuses

        p = self.Parametrization(tf.gather(s, indices1), tf.gather(t, indices1))
        p0 = tf.gather(measurements, indices1)

        q = self.Parametrization(tf.gather(s, indices2), tf.gather(t, indices2))
        q0 = tf.gather(measurements, indices2)

        num = q.euclideanDistance(p.projection(p0)) + p.euclideanDistance(q.projection(q0))
        den = tf.norm(p.projection(p0) - q.projection(q0), ord='euclidean', axis=1)

        return tf.concat([dist, self.lambd * num / den], axis=0)

    def getVectorJacobianProduct(self, x, v):
        func = self.getFunction(x)
        return tf.gradients(func, x, grad_ys=v)

    def getJacobianVectorProduct(self, x, u):
        # https://j-towns.github.io/2017/06/12/A-new-trick.html
        func = self.getFunction(x)
        v = tf.ones_like(func)
        vjp = tf.gradients(func, x, grad_ys=v)
        return tf.gradients(vjp, v, grad_ys=u)

    def getTangentLinesPoints(self, x):
        s,t = tf.split(x, num_or_size_splits=2)
        s = tf.reshape(s, [-1, self.numDimensions])
        t = tf.reshape(t, [-1, self.numDimensions])
        return (s,t)

    def getProjections(self, x):
        s,t = tf.split(x, num_or_size_splits=2)
        s = tf.reshape(s, [-1, self.numDimensions])
        t = tf.reshape(t, [-1, self.numDimensions])
        return self.Parametrization(s, t).projection(self.measurements)

    class Parametrization:
        def __init__(self, s, t):
            self.s = s
            self.t = t

        @staticmethod
        def numParams(numDimensions):
            return 2 * numDimensions

        def projection(self, x):
            s = self.s
            sMinusT = s - self.t
            num = tf.einsum('ij,ij->i', s - x, sMinusT)
            den = tf.einsum('ij,ij->i', sMinusT, sMinusT)
            lambd = num / den
            return s - tf.einsum('i,ij->ij', lambd, sMinusT)

        def euclideanDistance(self, x):
            return tf.norm(x - self.projection(x), ord='euclidean', axis=1)

        def euclideanDistanceSq(self, x):
            xMinusProj = x - self.projection(x)
            return tf.einsum('ij,ij->i', xMinusProj, xMinusProj)

class InexactLevenbergMarquardtMinimizer:
    def __init__(self, builder):
        self.builder = builder

    def minimize(self, dampmin=0.01, E=2.0, D=0.5, pi1=0.05, pi2=0.75, tolx=0.0001, tolf=0.0001, tolg=0.0001):
        builder = self.builder

        xArg = builder.getVariablePlaceholder(tf.float64)
        func = builder.getFunction(xArg)

        sess = tf.Session()

        init = tf.global_variables_initializer()
        sess.run(init)

        # 0:
        x = builder.getVariableValue()

        f = sess.run(func, {xArg: x})

        uArg = tf.placeholder(tf.float64, shape=x.shape)
        vArg = tf.placeholder(tf.float64, shape=f.shape)

        vjp = builder.getVectorJacobianProduct(xArg, vArg)
        jvp = builder.getJacobianVectorProduct(xArg, uArg)

        getJacobian = lambda x: LinearOperator(shape=(f.shape[0], x.shape[0]), matvec=lambda u: sess.run(jvp, {xArg : x, uArg : u}), rmatvec=lambda v: sess.run(vjp, {xArg : x, vArg : v}))

        k = 0
        damp = 1

        while True:
            jac = getJacobian(x)
            tau = InexactLevenbergMarquardtMinimizer.getKthEta(k) * linalg.norm(jac.rmatvec(f))

            y,istop,itn,r1norm,r2norm,anorm,acond,arnorm,xnorm,var = lsqr(jac, -f, damp, show=True)
   
            xnew = x + y
            fnew = sess.run(func, {xArg: xnew})
            F = np.einsum('i,i->', f, f)
            Fnew = np.einsum('i,i->', fnew, fnew)
            
            print('fnorm : {0:.3f}'.format(F))

            rho = (F - Fnew) / (F - r2norm * r2norm)

            if rho < pi1:
                if damp == 0:
                    damp = dampmin
                else:
                    damp = E * damp
            else:
                xconv = linalg.norm(y, ord=np.inf) / (linalg.norm(xnew, ord=np.inf) + linalg.norm(x, ord=np.inf)) <= tolx
                fconv = Fnew <= tolf
                gconv = linalg.norm(jac.rmatvec(f)) <= 0.5 * tolg

                x = xnew
                f = fnew

                if xconv or fconv or gconv:
                    print('fnorm : {0:.3f}'.format(Fnew))
                    break

                k = k + 1
                
                if rho > pi2:
                    damp = D * damp
                if damp < dampmin:
                    damp = 0

        s,t = builder.getTangentLinesPoints(x)
        proj = builder.getProjections(x)

        return sess.run(s), sess.run(t), sess.run(proj)

    @staticmethod
    def getKthEta(k):
        return 0.5