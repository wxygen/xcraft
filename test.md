$$
\begin{aligned}
\mathscr{F}(\mathbf{X}, \mathbf{Y}, \mathbf{Z}) = &\min_{\mathbf{U} \in \mathbb{C}^{m \times n}} \max_{\mathbf{V} \in \mathbb{C}^{p \times q}} \left\{ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \text{Tr} \left[ \mathbf{U}^\dagger \mathbf{V} \mathbf{X} \mathbf{Y}^\top \mathbf{Z} \right] e^{-\frac{1}{2}\|\mathbf{W}\|_F^2} d\mathbf{W} \right. \\
&+ \sum_{k=1}^{\infty} \frac{(-1)^k}{k!} \oint_{\gamma} \frac{\det(\mathbf{A} - z\mathbf{I})}{\prod_{j=1}^m (z - \lambda_j(\mathbf{B}))} dz \\
&+ \lim_{\epsilon \to 0^+} \frac{1}{\pi} \int_{\mathbb{R}^{n}} \frac{\Im\left\langle \psi \left| \mathbf{H} \right| \psi \right\rangle}{\omega - E_0 + i\epsilon} d\omega \\
&+ \sup_{\mathbf{P} \succeq 0} \inf_{\mathbf{Q} \succ 0} \left\{ \log \frac{\det(\mathbf{P} + \mathbf{Q})}{\det(\mathbf{P})\det(\mathbf{Q})} + \text{Tr} \left[ \mathbf{P}^{-1} \mathbf{Q} \mathbf{R} \mathbf{S}^\top \right] \right\} \\
&+ \prod_{i=1}^n \prod_{j=1}^m \left( \sum_{k=1}^p \mathbf{X}_{ijk} \mathbf{Y}_{kji} + \oint_{\partial \Omega} \mathbf{F}(\mathbf{r}) \cdot d\mathbf{r} \right) \\
&+ \bigcap_{\alpha \in \mathcal{A}} \bigcup_{\beta \in \mathcal{B}} \left\{ \mathbf{x} \in \mathbb{R}^n : \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2 \leq \epsilon \right\} \\
&+ \sum_{\substack{S \subseteq \{1,\dots,n\} \\ |S| = k}} \left( \prod_{i \in S} \lambda_i(\mathbf{C}) \right) \exp\left( -\frac{1}{2} \mathbf{x}_S^\top \mathbf{\Sigma}^{-1} \mathbf{x}_S \right) \\
&+ \int_{\mathcal{M}} \left\langle \nabla f, \nabla g \right\rangle dV + \oint_{\partial \mathcal{M}} f \frac{\partial g}{\partial n} dS \\
&+ \min_{\mathbf{\Theta} \in \text{SO}(n)} \max_{\mathbf{\Phi} \in \text{U}(m)} \left\| \mathbf{\Theta} \mathbf{X} \mathbf{\Phi}^\dagger - \mathbf{Y} \right\|_{\text{F}}^2 \\
&+ \sum_{d=0}^{\infty} \int_{\overline{\mathcal{M}}_{g,n}} \psi_1^{d_1} \cdots \psi_n^{d_n} \cap [\overline{\mathcal{M}}_{g,n}]^{\text{vir}} \\
&+ \sup_{f \in \mathcal{F}} \inf_{g \in \mathcal{G}} \left\{ \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \ell(f(x), g(y)) \right] + \lambda \|f\|_{\mathcal{H}}^2 \right\} \\
&+ \prod_{v \in V} \sum_{c \in C} \prod_{(u,v) \in E} \mathbf{1}_{c(u) \neq c(v)} \exp\left( -\beta H(\mathbf{c}) \right) \\
&+ \lim_{N \to \infty} \frac{1}{N} \log \int e^{-\beta H_N(\sigma)} d\sigma \\
&+ \sum_{\pi \in S_n} \text{sgn}(\pi) \prod_{i=1}^n \mathbf{A}_{i,\pi(i)} \oint_{\gamma} \frac{f(z)}{(z-a)^{n+1}} dz \\
&+ \min_{\mathbf{W}_1, \mathbf{W}_2} \max_{\mathbf{D}} \left\{ \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))] \right\} \\
&+ \int_{\text{Conf}_N(\mathbb{R}^2)} \prod_{i<j} |z_i - z_j|^\beta e^{-\frac{\beta}{4} \sum |z_i|^2} d^2 z_1 \cdots d^2 z_N \\
&+ \sum_{k=1}^K \int_{\Theta_k} p(x|\theta_k) \pi(\theta_k) d\theta_k \prod_{j\neq k} \mathbb{P}(\theta_j \in \Theta_j) \\
&+ \sup_{\mathbf{M} \in \mathcal{C}} \inf_{\mathbf{N} \in \mathcal{D}} \left\{ \text{Tr}[\mathbf{M}\mathbf{N}] + \log \det(\mathbf{I} + \mathbf{M} + \mathbf{N}) \right\} \\
&+ \prod_{i=1}^\infty \left( 1 + \frac{z}{i} \right) e^{-\frac{z}{i}} \Gamma(1+z) \zeta(1+z) \\
&+ \sum_{n=1}^\infty \frac{\chi(n)}{n^s} \prod_{p \mid n} \left(1 - \frac{1}{p^s}\right) L(s, \chi) \\
&+ \min_{\mathbf{U}} \max_{\mathbf{V}} \left\{ \left\| \mathbf{U} \mathbf{V}^\top - \mathbf{X} \right\|_* + \lambda \|\mathbf{U}\|_{2,1} + \mu \|\mathbf{V}\|_{1,2} \right\} \\
&+ \int_{\text{Gr}(k,n)} \sigma_\lambda \cdot \sigma_\mu \cup \sigma_\nu \cap [\text{Gr}(k,n)] \\
&+ \sum_{\mathbf{m} \in \mathbb{Z}^n} e^{-\mathbf{m}^\top \mathbf{A} \mathbf{m} + \mathbf{b}^\top \mathbf{m}} \prod_{i=1}^n \theta_3(z_i, q_i) \\
&+ \limsup_{n \to \infty} \frac{1}{n} \log \mathbb{P} \left( \frac{1}{n} \sum_{i=1}^n X_i \in A \right) \\
&+ \min_{\rho \in \mathcal{D}(\mathcal{H})} \max_{M \in \text{POVM}} I(X;Y) + \lambda S(\rho) \\
&+ \int_{\mathcal{A}/\mathcal{G}} e^{-S_{\text{YM}}[A]} \mathcal{D}A \prod_{x \in M} \delta(G(A)(x)) \\
&+ \sum_{\Gamma} \frac{1}{|\text{Aut}(\Gamma)|} \int_{\overline{\mathcal{M}}_{g,n}} \omega_\Gamma \cap [\overline{\mathcal{M}}_{g,n}] \\
&+ \sup_{f \in BL_1} \left\{ \int f d\mu - \int f d\nu \right\} + \lambda W_2^2(\mu, \nu) \\
&+ \prod_{p \in \text{Spec}(\mathbf{D})} \left( 1 - \frac{s}{p} \right)^{-1} \exp\left( \sum_{m=1}^\infty \frac{N_p^m}{m} p^{-ms} \right) \\
&+ \min_{\mathbf{T}} \max_{\mathbf{S}} \left\{ \text{Tr}[\mathbf{T}^\top \mathbf{S}] + \|\mathbf{T} - \mathbf{S}\|_{\text{HS}}^2 \right\} \\
&+ \int_{\text{Lag}(M)} \mathcal{F}(\mathcal{L}) \wedge \star \mathcal{F}(\mathcal{L}) \exp\left( -\int_M R \right) \\
&+ \sum_{\chi \in \widehat{G}} \dim V_\chi \oint_{\gamma} \frac{\chi(g)}{\det(1 - g)} dg \\
&+ \lim_{\Lambda \to \infty} \int_{|p|<\Lambda} \frac{d^4p}{(2\pi)^4} \frac{\text{Tr}[\gamma^\mu \gamma^\nu \gamma^\rho \gamma^\sigma]}{p^2 - m^2} \\
&+ \min_{\pi} \max_{s \in \mathcal{S}} \left\{ R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s,\pi(s)) V(s') \right\} \\
&+ \int_{\mathcal{X}} \left( \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n f(X_i) - \mathbb{E}[f(X)] \right) d\mathbb{P} \\
&+ \sum_{n=1}^\infty \frac{\tau(n)}{n^s} \prod_{p} \left( 1 - \frac{\tau(p)}{p^s} + \frac{p^{11}}{p^{2s}} \right)^{-1} \\
&+ \min_{\mathbf{W}} \max_{\mathbf{H}} \left\{ \text{Tr}[\mathbf{W}^\top \mathbf{H} \mathbf{W}] + \lambda \|\mathbf{W}\|_{2,1} + \mu \|\mathbf{H}\|_* \right\} \\
&+ \oint_{\gamma} \frac{\prod_{i=1}^n (z - a_i)^{m_i}}{\prod_{j=1}^p (z - b_j)^{n_j}} dz \times \frac{1}{2\pi i} \int_{\Gamma} f(w) dw \\
&+ \sum_{k=0}^\infty \frac{(-1)^k}{(2k+1)!} z^{2k+1} \prod_{j=1}^k \frac{1}{(2j-1)!!} \\
&+ \lim_{t \to \infty} \frac{1}{t} \log \mathbb{E} \left[ \exp\left( \int_0^t V(X_s) ds \right) \right] \\
&+ \min_{\mathbf{P}} \max_{\mathbf{Q}} \left\{ \text{Tr}[\mathbf{P} \mathbf{Q}] + \log \det(\mathbf{I} + \mathbf{P} + \mathbf{Q}) \right\} \\
&+ \int_{\text{Bun}_G(X)} \mathcal{F}(\mathcal{E}) \wedge \star \mathcal{F}(\mathcal{E}) \exp\left( -\int_X \text{ch}(\mathcal{E}) \right) \\
&+ \sum_{\rho \in \widehat{G}} \dim \rho \oint_{\gamma} \frac{\chi_\rho(g)}{\det(1 - \text{Ad}(g))} dg \\
&+ \lim_{\epsilon \to 0} \frac{1}{\epsilon} \left( \int_{\mathbb{R}^n} f(x) e^{-\frac{\|x\|^2}{2\epsilon}} dx - (2\pi\epsilon)^{n/2} f(0) \right) \\
&+ \min_{\theta \in \Theta} \max_{x \in \mathcal{X}} \left\{ \ell(\theta, x) + \lambda R(\theta) + \mu \Omega(x) \right\} \\
&+ \int_{\mathcal{M}} \text{Tr}[\mathcal{R} \wedge \mathcal{R}] \wedge \star \text{Tr}[\mathcal{F} \wedge \mathcal{F}] \\
&+ \sum_{n=1}^\infty \frac{a_n}{n^s} \prod_{p} \left( 1 - \frac{a_p}{p^s} + \frac{\psi(p)}{p^{2s}} \right)^{-1} L(s, \psi) \\
&+ \sup_{f \in \mathcal{F}} \inf_{g \in \mathcal{G}} \left\{ \mathbb{E}[\ell(f,g)] + \lambda \|f\|_{\mathcal{H}}^2 + \mu \|g\|_{\mathcal{H}}^2 \right\} \\
&+ \prod_{v \in V} \sum_{\sigma_v \in \Sigma} \prod_{(u,v) \in E} \exp\left( -\beta J_{uv} \sigma_u \sigma_v - h_v \sigma_v \right) \\
&+ \lim_{N \to \infty} \frac{1}{N} \log \int \exp\left( -\beta H_N(\sigma) + h \cdot \sigma \right) d\sigma \\
&+ \sum_{\pi \in S_n} \varepsilon(\pi) \prod_{i=1}^n \mathbf{A}_{i,\pi(i)} \oint_{\gamma} \frac{f(z)}{\prod_{j=1}^m (z - a_j)^{k_j}} dz \\
&+ \min_{\mathbf{W}} \max_{\mathbf{D}} \left\{ \mathbb{E}[\log D(x)] - \mathbb{E}[\log(1-D(G(z)))] \right\} \\
&+ \int_{\text{Sym}^N(M)} \prod_{i<j} d(z_i, z_j)^\beta e^{-\beta U(\mathbf{z})} d\text{vol}(\mathbf{z}) \\
&+ \sum_{k=1}^K \int_{\Theta_k} p(x|\theta_k) \pi(\theta_k) d\theta_k \prod_{j\neq k} F(\theta_j) d\theta_j \\
&+ \sup_{\mathbf{M}} \inf_{\mathbf{N}} \left\{ \text{Tr}[\mathbf{M}\mathbf{N}] + \log \det(\mathbf{I} + \mathbf{M}\mathbf{N}) \right\} \\
&+ \prod_{n=1}^\infty (1 - q^n)^{24} \eta(\tau)^{24} \Delta(\tau) \\
&+ \sum_{\chi} \frac{L(s, \chi)}{L(2s, \chi^2)} \prod_{p} \left( 1 - \frac{\chi(p)}{p^s} \right)^{-1} \\
&+ \min_{\mathbf{U}} \max_{\mathbf{V}} \left\{ \|\mathbf{U} \mathbf{V}^\top - \mathbf{X}\|_F^2 + \lambda \|\mathbf{U}\|_* + \mu \|\mathbf{V}\|_* \right\} \\
&+ \int_{\mathcal{M}_{g,n}} \psi_1^{a_1} \cdots \psi_n^{a_n} \kappa_1^{b_1} \cdots \kappa_m^{b_m} \cap [\mathcal{M}_{g,n}]^{\text{vir}} \\
&+ \sum_{\mathbf{m} \in \mathbb{Z}^n} e^{-\mathbf{m}^\top \mathbf{\Sigma} \mathbf{m} + \mathbf{\mu}^\top \mathbf{m}} \prod_{i=1}^n \theta(z_i | \tau_i) \\
&+ \liminf_{n \to \infty} \frac{1}{n} \log \mathbb{P} \left( \bigcap_{i=1}^n \{ X_i \in A_i \} \right) \\
&+ \min_{\rho} \max_{\Lambda} \left\{ S(\rho \| \sigma) + \lambda \text{Tr}[\rho \log \rho] + \mu \text{Tr}[\sigma \log \sigma] \right\} \\
&+ \int_{\mathcal{A}} e^{-S_{\text{CS}}[A]} \mathcal{D}A \prod_{x} \delta(F_A(x)) \\
&+ \sum_{\Gamma} \frac{1}{|\text{Aut}(\Gamma)|} \int_{\mathcal{M}_{g,n}} \omega_\Gamma \wedge \star \omega_\Gamma \\
&+ \sup_{f} \left\{ \int f d\mu - \int f d\nu \right\} + \lambda \text{TV}(\mu, \nu) + \mu \text{KL}(\mu \| \nu) \\
&+ \prod_{p} \left( 1 - \frac{\alpha_p}{p^s} \right)^{-1} \left( 1 - \frac{\beta_p}{p^s} \right)^{-1} L(s, f) L(s, g) \\
&+ \min_{\mathbf{T}} \max_{\mathbf{S}} \left\{ \text{Tr}[\mathbf{T}^\top \mathbf{S} \mathbf{T}] + \|\mathbf{T} - \mathbf{S}\|_F^2 + \lambda \|\mathbf{T}\|_* \right\} \\
&+ \int_{\text{Loc}(M)} \mathcal{F}(\nabla) \wedge \star \mathcal{F}(\nabla) \exp\left( -\int_M \text{scal} \right) \\
&+ \sum_{\chi} \dim \rho_\chi \oint_{\gamma} \frac{\chi(g)}{\det(1 - \rho(g))} dg \\
&+ \lim_{\Lambda \to \infty} \int \frac{d^d p}{(2\pi)^d} \frac{\text{Tr}[\Gamma^{\mu_1} \cdots \Gamma^{\mu_n}]}{(p^2 - m^2)^k} \\
&+ \min_{\pi} \max_{s} \left\{ Q(s, \pi(s)) + \gamma \mathbb{E}[V(s')] + \lambda H(\pi(\cdot|s)) \right\} \\
&+ \int_{\mathcal{X}} \left( \mathbb{E}[f] - \frac{1}{n} \sum f(X_i) \right)^2 d\mathbb{P} + \lambda \|f\|_{\mathcal{H}}^2 \\
&+ \sum_{n=1}^\infty \frac{\sigma_k(n)}{n^s} \prod_{p} \left( 1 - \frac{\sigma_k(p)}{p^s} + \frac{p^{2k}}{p^{2s}} \right)^{-1} \\
&+ \min_{\mathbf{W}} \max_{\mathbf{H}} \left\{ \text{Tr}[\mathbf{W}^\top \mathbf{H} \mathbf{W}] + \lambda \|\mathbf{W}\|_{2,1} + \mu \|\mathbf{H}\|_F^2 \right\} \\
&+ \oint_{\gamma} \frac{\prod (z - a_i)^{m_i}}{\prod (z - b_j)^{n_j}} \frac{f(z)}{g(z)} dz \times \frac{1}{2\pi i} \int_{\Gamma} h(w) dw \\
&+ \sum_{k=0}^\infty \frac{(-1)^k}{(2k)!} z^{2k} \prod_{j=1}^k \frac{1}{(2j)!!} B_{2k} \\
&+ \lim_{t \to 0} \frac{1}{t} \left( \mathbb{E}[f(X_t)] - f(X_0) \right) \\
&+ \min_{\mathbf{P}} \max_{\mathbf{Q}} \left\{ \text{Tr}[\mathbf{P} \mathbf{Q} \mathbf{P}^\top] + \log \det(\mathbf{I} + \mathbf{P} \mathbf{Q}) \right\} \\
&+ \int_{\text{Coh}(X)} \text{ch}(\mathcal{E}) \wedge \text{td}(X) \exp\left( -\int_X c_1(\mathcal{E}) \right) \\
&+ \sum_{\rho} \dim \rho \oint_{\gamma} \frac{\chi_\rho(g)}{\det(1 - \text{Ad}^*(g))} dg \\
&+ \lim_{\epsilon \to 0} \frac{1}{\epsilon^2} \left( \int f(x) e^{-\frac{\|x\|^2}{2\epsilon}} dx - (2\pi\epsilon)^{n/2} \sum_{|\alpha| \leq k} \frac{D^\alpha f(0)}{\alpha!} \right) \\
&+ \min_{\theta} \max_{x} \left\{ L(\theta, x) + \lambda \|\theta\|_1 + \mu \|x\|_\infty + \nu R(\theta, x) \right\} \\
&+ \int_{\mathcal{M}} \text{Tr}[\mathcal{R} \wedge \star \mathcal{R}] \wedge \text{Tr}[\mathcal{F} \wedge \star \mathcal{F}] \\
&+ \sum_{n=1}^\infty \frac{\lambda_f(n)}{n^s} \prod_{p} \left( 1 - \frac{\lambda_f(p)}{p^s} + \frac{\psi(p)}{p^{2s}} \right)^{-1} \\
&+ \sup_{f} \inf_{g} \left\{ \mathbb{E}[\ell(f,g)] + \lambda \|f\|^2 + \mu \|g\|^2 + \nu \text{Cov}(f,g) \right\} \\
&+ \prod_{v} \sum_{\sigma_v} \prod_{(u,v)} \exp\left( -\beta J_{uv} \sigma_u \sigma_v - h_u \sigma_u - h_v \sigma_v \right) \\
&+ \lim_{N \to \infty} \frac{1}{N} \log \int \exp\left( -\beta H_N(\sigma) + \mathbf{h} \cdot \sigma + \mathbf{J} \cdot \sigma \sigma^\top \right) d\sigma \\
&+ \sum_{\pi} \varepsilon(\pi) \prod_{i} \mathbf{A}_{i,\pi(i)} \oint_{\gamma} \frac{f(z)}{\prod (z - a_j)} \frac{g(z)}{\prod (z - b_k)} dz \\
&+ \min_{\mathbf{W}} \max_{\mathbf{D}} \left\{ \mathbb{E}[\log D] + \mathbb{E}[\log(1-D)] + \lambda \|\mathbf{W}\|^2 + \mu \|\mathbf{D}\|^2 \right\} \\
&+ \int_{\text{Conf}_N} \prod_{i<j} |z_i - z_j|^{2\beta} e^{-\beta \sum V(z_i)} d^2 z_1 \cdots d^2 z_N \\
&+ \sum_{k=1}^K \int_{\Theta_k} p(x|\theta_k) \pi(\theta_k) d\theta_k \prod_{j} \mathbb{P}(\theta_j \in A_j) F_j(\theta_j) \\
&+ \sup_{\mathbf{M}} \inf_{\mathbf{N}} \left\{ \text{Tr}[\mathbf{M}\mathbf{N}\mathbf{M}^\top] + \log \det(\mathbf{I} + \mathbf{M}\mathbf{N}) + \lambda \|\mathbf{M}\|_F^2 \right\} \\
&+ \prod_{n=1}^\infty (1 - q^n)^{24} (1 - q^{2n})^{24} \eta(\tau)^{48} \Delta(\tau)^2 \\
&+ \sum_{\chi} \frac{L(s, \chi) L(s, \chi^2)}{L(2s, \chi^4)} \prod_{p} \left( 1 - \frac{\chi(p)^2}{p^s} \right)^{-1} \\
&+ \min_{\mathbf{U}} \max_{\mathbf{V}} \left\{ \|\mathbf{U} \mathbf{V}^\top - \mathbf{X}\|^2 + \lambda \|\mathbf{U}\|_{2,1} + \mu \|\mathbf{V}\|_{1,2} + \nu \|\mathbf{U} \mathbf{V}^\top\|_* \right\} \\
&+ \int_{\overline{\mathcal{M}}_{g,n}} \psi_1^{d_1} \cdots \psi_n^{d_n} \kappa_1^{e_1} \cdots \kappa_m^{e_m} \lambda_1^{f_1} \cdots \lambda_k^{f_k} \cap [\overline{\mathcal{M}}_{g,n}] \\
&+ \sum_{\mathbf{m} \in \mathbb{Z}^n} e^{-\mathbf{m}^\top \mathbf{\Sigma}^{-1} \mathbf{m} + \mathbf{\mu}^\top \mathbf{m}} \prod_{i=1}^n \theta(z_i | \tau) \vartheta(z_i | \tau) \\
&+ \lim_{n \to \infty} \frac{1}{n} \log \mathbb{P} \left( \bigcap_{i=1}^n \bigcup_{j=1}^m \{ X_{ij} \in A_{ij} \} \right) \\
&+ \min_{\rho} \max_{\Lambda} \left\{ I(X;Y) + \lambda S(\rho) + \mu \text{Tr}[\rho \log \rho] + \nu \text{Tr}[\sigma \log \sigma] \right\} \\
&+ \int_{\mathcal{A}} e^{-S[A]} \mathcal{D}A \prod_{x} \delta(\mathcal{G}(A)(x)) \det(\mathcal{M}(A)) \\
&+ \sum_{\Gamma} \frac{1}{|\text{Aut}(\Gamma)|} \int_{\mathcal{M}_{g,n}} \omega_\Gamma \wedge \overline{\omega_\Gamma} \wedge \Phi_\Gamma \\
&+ \sup_{f} \left\{ \int f d\mu - \int f d\nu \right\} + \sum_{k=1}^\infty \lambda_k W_k^k(\mu, \nu) + \text{KL}(\mu \| \nu) \\
&+ \prod_{p} \left( 1 - \frac{\alpha_p}{p^s} \right)^{-1} \left( 1 - \frac{\beta_p}{p^s} \right)^{-1} \left( 1 - \frac{\gamma_p}{p^s} \right)^{-1} L(s,f) L(s,g) L(s,h) \\
&+ \min_{\mathbf{T}} \max_{\mathbf{S}} \left\{ \text{Tr}[\mathbf{T}^\top \mathbf{S} \mathbf{T} \mathbf{S}^\top] + \|\mathbf{T} - \mathbf{S}\|_F^2 + \lambda \|\mathbf{T}\|_* + \mu \|\mathbf{S}\|_* \right\}
\end{aligned}
$$