import numpy as np
from scipy.stats import chi2


class vehicle_simulation:
    def __init__(self, vehicles=None):
        self.time = 0
        self.delta_t = 1.0

        if vehicles is None:
            self.vehicles = []
        else:
            self.vehicles = vehicles

    def add_vehicle(self, vehicle):
        # Typo fix: should be self.vehicles, not self.vehicle
        self.vehicles.append(vehicle)

    def is_collision(self, confidence):
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                vehicleA = self.vehicles[i]
                vehicleB = self.vehicles[j]

                A_x, A_y, A_v, A_yaw = vehicleA.pred_next_state
                B_x, B_y, B_v, B_yaw = vehicleB.pred_next_state

                rel_pos = np.array([A_x - B_x, A_y - B_y, 0])
                rel_vel = np.array([
                    A_v * np.cos(A_yaw) - B_v * np.cos(B_yaw),
                    A_v * np.sin(A_yaw) - B_v * np.sin(B_yaw),
                    0
                ])

                if np.linalg.norm(rel_vel) < 1e-6:
                    # vehicles have same velocity - check distance-based collision
                    rel_P = vehicleA.predicted_P + vehicleB.predicted_P
                    P_pos = rel_P[:2, :2]  # position covariance only
                    try:
                        d_square = rel_pos[:2].T @ np.linalg.inv(P_pos) @ rel_pos[:2]
                        crit = chi2.ppf(confidence, df=2)
                        if d_square < crit:
                            return True, i, j
                    except np.linalg.LinAlgError:
                        # fallback to simple distance if matrix singular
                        if np.linalg.norm(rel_pos[:2]) < 5.0:  # 5m threshold
                            return True, i, j
                    continue

                rel_vel_normal = rel_vel / np.linalg.norm(rel_vel)
                rel_P = vehicleA.predicted_P + vehicleB.predicted_P

                # Construct orthogonal basis for b-plane
                first_orth_basis = np.array([-rel_vel[1], rel_vel[0], 0])
                first_orth_basis /= np.linalg.norm(first_orth_basis)

                second_orth_basis = np.cross(first_orth_basis, rel_vel_normal)
                second_orth_basis /= np.linalg.norm(second_orth_basis)

                # Project miss distance vector onto plane
                b = rel_pos - np.dot(rel_pos, rel_vel_normal) * rel_vel_normal

                # Transform covariance into b-plane coordinates
                H_B = np.vstack([first_orth_basis, second_orth_basis])
                # reduce P to 2D subspace
                P_B = H_B @ rel_P[:3, :3] @ H_B.T

                # Mahalanobis distance
                d_square = b[:2].T @ np.linalg.inv(P_B) @ b[:2]

                crit = chi2.ppf(confidence, df=2)

                # Collision if Mahalanobis distance within α-ellipse
                if d_square < crit:
                    return True, i, j

        return False, None, None

    def time_step(self, control_inputs):
        for i in range(len(self.vehicles)):
            self.vehicles[i].update_external(control_inputs[i], self.delta_t)
        for i in range(len(self.vehicles)):
            self.vehicles[i].ukf_predict(self.delta_t)

        is_collision, index_1, index_2 = self.is_collision(0.95)
        if is_collision:
            return False, index_1, index_2

        self.time += self.delta_t
        return True, None, None


class vehicle:
    def __init__(self, L, control_input=None, external_state=None, internal_state=None,
                 P=None, Q=None, R=None, alpha=1e-3, beta=2, kappa=0):
        self.L = L

        if control_input is None:
            self.control_input = np.array([0.0, 0.0])
        else:
            self.control_input = np.array(control_input, dtype=float)

        if external_state is None:
            self.external_state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.external_state = np.array(external_state, dtype=float)

        if internal_state is None:
            self.internal_state = self.external_state.copy()
        else:
            self.internal_state = np.array(internal_state, dtype=float)

        self.P = np.diag([0.1**2, 0.1**2, 0.5**2, 0.05**2]) if P is None else np.array(P, dtype=float)
        self.Q = np.diag([0.01**2, 0.01**2, 0.1**2, 0.01**2]) if Q is None else np.array(Q, dtype=float)
        self.R = np.diag([0.1**2, 0.1**2, 0.01**2]) if R is None else np.array(R, dtype=float)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = 4  # state dimension
        self.lambda_val = alpha**2 * (self.n + kappa) - self.n

        # UKF Weights
        self.weights_m = np.zeros(2*self.n + 1)
        self.weights_c = np.zeros(2*self.n + 1)
        self.weights_m[0] = self.lambda_val / (self.n + self.lambda_val)
        self.weights_c[0] = self.lambda_val / (self.n + self.lambda_val) + (1 - alpha**2 + beta)
        for i in range(1, 2*self.n + 1):
            self.weights_m[i] = 1 / (2 * (self.n + self.lambda_val))
            self.weights_c[i] = 1 / (2 * (self.n + self.lambda_val))

        # temporary prediction variables
        self.pred_next_state = np.zeros(self.n)
        self.predicted_P = np.zeros((self.n, self.n))

    def measurement(self, state):
        x, y, _, phi = state
        mean = np.zeros(3)
        noise = np.random.multivariate_normal(mean, self.R)
        return np.array([x, y, phi]) + noise

    def bicycle_update(self, state, delta_t):
        x, y, v, phi = state
        a, steering_angle = self.control_input
        x_new = x + v * np.cos(phi) * delta_t
        y_new = y + v * np.sin(phi) * delta_t
        phi_new = phi + (v / self.L) * np.tan(steering_angle) * delta_t
        v_new = v + a * delta_t
        return np.array([x_new, y_new, v_new, phi_new])

    def ukf_predict(self, delta_t):
        n = self.n
        lambda_val = self.lambda_val
        sqrt_P = np.linalg.cholesky((n + lambda_val) * self.P)

        sigma_points = np.zeros((n, 2*n + 1))
        sigma_points[:, 0] = self.internal_state
        for i in range(n):
            sigma_points[:, i+1] = self.internal_state + sqrt_P[:, i]
            sigma_points[:, i+1+n] = self.internal_state - sqrt_P[:, i]

        sigma_nexts = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            sigma_nexts[:, i] = self.bicycle_update(sigma_points[:, i], delta_t)

        sigma_next_mean = np.sum(sigma_nexts * self.weights_m.reshape(1, -1), axis=1)

        predicted_P = np.zeros((n, n))
        for i in range(2*n + 1):
            diff = (sigma_nexts[:, i] - sigma_next_mean).reshape(-1, 1)
            predicted_P += self.weights_c[i] * diff @ diff.T
        predicted_P += self.Q

        self.pred_next_state = sigma_next_mean
        self.predicted_P = predicted_P

        return sigma_points, sigma_nexts, sigma_next_mean, predicted_P

    def ukf_update(self, delta_t):
        n = self.n
        m = 3

        sigma_points, sigma_nexts, sigma_pred, P_pred = self.ukf_predict(delta_t)

        Z = np.zeros((m, 2*n + 1))
        for i in range(2*n + 1):
            x, y, _, phi = sigma_nexts[:, i]
            Z[:, i] = np.array([x, y, phi])

        z_pred = np.sum(Z * self.weights_m.reshape(1, -1), axis=1)

        S = np.zeros((m, m))
        for i in range(2*n + 1):
            dz = (Z[:, i] - z_pred).reshape(-1, 1)
            S += self.weights_c[i] * dz @ dz.T
        S += self.R

        Pxz = np.zeros((n, m))
        for i in range(2*n + 1):
            dx = (sigma_nexts[:, i] - sigma_pred).reshape(-1, 1)
            dz = (Z[:, i] - z_pred).reshape(-1, 1)
            Pxz += self.weights_c[i] * dx @ dz.T

        K = Pxz @ np.linalg.inv(S)

        z_actual = self.measurement(self.external_state)
        self.internal_state = sigma_pred + K @ (z_actual - z_pred)
        self.P = P_pred - K @ S @ K.T

    def update_external(self, control_input, delta_t):
        process_noise = np.random.multivariate_normal(np.zeros(4), self.Q)
        self.external_state = self.bicycle_update(self.external_state, delta_t) + process_noise
        self.control_input = control_input
        self.ukf_update(delta_t)
        
# --- Setup vehicles ---
# Two cars on a near-collision path
carA = vehicle(
    L=2.5,
    external_state=[0, 0, 10, 0],  # (x, y, v, phi)
    control_input=[0.0, 0.0],      # a, delta
)

carB = vehicle(
    L=2.5,
    external_state=[50, 1, 8, np.pi],  # Coming toward carA
    control_input=[0.0, 0.0],
)

# --- Setup simulation ---
sim = vehicle_simulation([carA, carB])

# --- Run Simulation ---
steps = 10
for t in range(steps):
    print(f"\n⏱️ Time step {t}:")
    for i, v in enumerate(sim.vehicles):
        print(f"Vehicle {i} external state before step: {v.external_state}")

    # Control inputs (both go straight for now)
    controls = [[0.0, 0.0], [0.0, 0.0]]

    # Run one time step
    result, idx1, idx2 = sim.time_step(controls)

    # Check for collisions
    if not result:
        print(f"⚠️ Collision predicted between vehicle {idx1} and vehicle {idx2}!")
        break

    for i, v in enumerate(sim.vehicles):
        print(f"Vehicle {i} external state after step: {v.external_state}")
        print(f"Vehicle {i} predicted state: {v.pred_next_state}")

print("\n✅ Simulation finished.")
