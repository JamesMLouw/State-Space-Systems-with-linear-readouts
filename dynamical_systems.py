from scipy.integrate import solve_ivp
import numpy as np

def iter_rk45(prev, t, h, f, fargs=None): # on wikipedia this is simply called the RK4 method.
    
    if fargs == None:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1)
        z3 = prev + (h/2)*f(t + 0.5*h, z2)
        z4 = prev + h*f(t + 0.5*h, z3)

        z = (h/6)*(f(t, z1) + 2*f(t + 0.5*h, z2) + 2*f(t + 0.5*h, z3) + f(t + h, z4))
        curr = prev + z
    
    else:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, fargs)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, fargs)
        z4 = prev + h*f(t + 0.5*h, z3, fargs)

        z = (h/6)*(f(t, z1, fargs) + 2*f(t + 0.5*h, z2, fargs) + 2*f(t + 0.5*h, z3, fargs) + f(t + h, z4, fargs))
        curr = prev + z
    
    return curr

class DS():
    def __init__(self, du_dt, n_dim, step, method='RK45'):
        self.du_dt = du_dt
        self.n_dim = n_dim
        self.step = step
        self.method = method

    def forward_step(self, y0, t0 = 0):
        return iter_rk45(y0, t0, self.step, self.du_dt)
        # return solve_ivp(self.du_dt, (t0, t0 + self.step), y0, method = self.method).y[:,-1] # I prefer to use our own RK45 in case this turns out too slow
        
    def forward(self, y0, T, t0 = 0):
        """
        Integrates forward over a batch for T time steps
        y0 : (batch, n_dim)
        """
        batch = y0.shape[0]
        y_sol = np.zeros((batch, T, n_dim))
        y_step = y0 # (batch, n_dim)
        t_step = t0
        for t in range(T):
            for i in range(batch):
                y = y_step[i,:].reshape(self.n_dim)
                y_step[i,:] = self.forward_step(y, t_step).reshape((1,self.n_dim))

            y_sol[:,t,:] = y_step.reshape((batch, 1, self.n_dim))
            t_step += self.step

        return y_sol
    

    def generate_ics(self, batch, y0 = None, sigma = 1, distribution = 'uniform', seed = None):
        np.random.seed(seed)

        y = np.array([y0 for _ in range(batch)])

        if distribution == "gaussian":
            y += np.random.normal(0, sigma, (batch, y0.shape[0]))
            
        elif distribution == "uniform":
            z += np.random.uniform(-sd, sd, (length, len(center)))

        else:
            print('No such pdf')

        return z