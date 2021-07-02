
import jax
import jax.numpy as jnp


rng = jax.random.PRNGKey(43)

def gradient_descent(A, b, iters=1000, tol=1e-8, debug=False):
    
    n = A.shape[0]

    x = jax.random.uniform(rng, shape=(n, 1))
    r = b - A @ x
    
    i = 0

    while i < iters and jnp.linalg.norm(A @ x - b) > tol:
        if debug:
            print( f"{i} th iter, err {jnp.linalg.norm(A @ x - b)}" )

        Ar = A @ r
        tau = r.T @ r / ( r.T @ Ar )
        x += tau * r
        r-= tau * Ar
        
        i += 1

    return x


def positive_definite(n):
    A = jax.random.uniform(rng, shape=(n, n))
    A = A @ A.T
    return A
            
n = 4
A = positive_definite(n)
b = jax.random.uniform(rng, shape=(n, 1))

x = gradient_descent(A, b)

assert( jnp.linalg.norm(A @ x - b) < 1e-2 )


