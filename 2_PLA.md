# PLA

## PLA

​	Well..., perceptron learning algorithm again. Because I've been very familiar with PLA on traditional linearly separable dataset, so I will briefly recap the PLA Model and pay more attention to 

1.  Why PLA must can stop.
2.  PLA on linearly inseparable dataset.

### PLA Model

#### basic component

-   input space $\mathcal{X}$: feature vector of length $d$

-   output space $\mathcal{Y}$:$\{-1,+1\}$
-   data $\mathcal{D}=\{(x_1,y_1),...,(x_N,y_N)\}$, where $x\in \mathcal{X}, y\in \mathcal{Y}$
-   hypothesis $\mathcal{H}$:  $h(\mathbf{x})=\operatorname{sign}\left(w^Tx\right)$, note $x$ is with bias (means it's dim is $d+1$ now)

#### Algorithm

Cycle on whole dataset, if $y_ih(x_i)<0$, let $w=w+y_ix_i$ until $y_ih(x_i)\geq 0$ for all sample. You may also set a learning rate $r$ and a batch $m$ and apply SGD. 

### Why PLA finally stop

​	So why the algorithm above must stop? I never thought about this before.

​	If the dataset is linearly separable, then there exists a $w_f$ which is the ideal parameter. Say, after $t$ circles update, out learned parameter become $w_t$. We compute the following formula:
$$
\frac{w_f}{||w_f||}*\frac{w_t}{||w_t||}
$$
​	Now suppose after $T$ times update, we get a $w_T$ that make formula above equal 1, which means that $w_T$ and $w_f$ are in the same direction, so we can say that we have find the ideal $w_f$, that is $w_T$.

​	If we initialize $w_t$ with 0 vector, after $T$ times update, $w_T=\sum_{t=1}^{T}y_tx_t$ where $y_t, x_t$ is the wrong classified sample at $t$-th update.

1.  we have $w_f*w_T=\sum_{t=1}^{T}y_tw_fx_t$, and $y_tw_fx_t > 0$ for all $t$ because $w_f$ is the perfect parameter we want to get. Therefore, $w_f*w_T=\sum_{t=1}^{T}y_tw_fx_t \geq T*min(y_tw_fx_t) \geq 0$. 
2.  According to property of vector, $||w_T||^2\leq T*max||y_tx_t||^2$, $y_t\in\{-1, +1\}$, so $||w_T||\leq max||x_t||$. *( you may also prove it by simplifying $||w_{t+1}||^2=||w_{t}+y_tx_t||^2$, )it's easy to understand, so I just skip this*, then $||w_T|| < \sqrt{T}max||x_t||$
3.  So we have $\frac{w_f}{||w_f||}*\frac{w_t}{||w_t||}\geq \frac{T*min(y_iw_fx_i)}{||w_f||*\sqrt{T}max||x_t||}=\sqrt{T}\frac{min(y_iw_fx_i)}{||w_f||*max||x_t||}$

Ok, if $T$ is big enough and make $\sqrt{T}\frac{min(y_iw_fx_i)}{||w_f||*max||x_t||}=1$, then we can say after T time update, $w_T$ must equal to $w_f$. So after no more than $\frac{||w_f||^2*max||x_t||^2}{min(y_iw_fx_i)^2}$ times steps, algorithm will stop.

### Pocket Algorithm

​	Then there comes another question that if the dataset is not linearly separable? After we always get some outliers due to noise or some other influence. Under this condition, we have two choice: 1. global optimal, 2. greedy.

​	If we want a global optimal algorithm, the problem becomes:
$$
\underset{\mathbf{w}}{\operatorname{argmin}} \sum_{n=1}^{N}\left[y_{n} \neq \operatorname{sign}\left(\mathbf{w}^{T} \mathbf{x}_{n}\right)\right]
$$
It's a NP-hard problem...., so let use greedy happily\^_\^.

Every time algorithm meets a wrong classified sample, it update the parameter in the same way as PLA do. But, after this, new algorithm apply the updated parameter on the whole dataset and observe whether new parameter works better than the old one. If so, we 'hold the new parameter in pocket'. After some steps, we let the algorithm stop and say the parameter in my pocket is just the result.  