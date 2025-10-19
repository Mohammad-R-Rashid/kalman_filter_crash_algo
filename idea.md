Okay, so the goal of the project is the predict car collisions based on principles of celestial mechanics. The principles include using a bicycle model with a ukf filter to predict positions at discrete time-steps and calculate an uncertainty ellipse, using b-planes to estimate probabilities of collisions, etc.


Nice — this is a fascinating cross-disciplinary idea (bringing b-plane methods from celestial mechanics into short-range vehicle collision prediction). Below I’ll lay out a practical, technically precise plan you can implement (and validate) right away:

Project plan (concept → implementation → validation)
1) Problem statement & high-level idea

Use a kinematic/dynamic bicycle (single-track) model for each vehicle, estimate state and state covariance online with an Unscented Kalman Filter (UKF), propagate the joint uncertainty to a target plane (the analog of a celestial b-plane) at discrete future time steps, then compute the probability that the relative miss vector falls inside a collision region (vehicle footprint). This borrows the geometric/uncertainty projection ideas used for asteroid close-approach Pc estimation onto a plane normal to the relative velocity. Background references: UKF & vehicle estimation, kinematic bicycle models, and b-plane / Pc methods in astrodynamics. 
PMC
+2
arXiv
+2

2) State, models and UKF setup
State vector (example)

For each vehicle 
i
i, choose a minimal state consistent with your sensor set (GPS/odometry/IMU/LiDAR):

xi=[x, y, v, ψ, ψ˙]⊤
x
i
	​

=[x,y,v,ψ,
ψ
˙
	​

]
⊤

where 
x,y
x,y are inertial positions, 
v
v speed, 
ψ
ψ heading (yaw), 
ψ˙
ψ
˙
	​

 yaw rate. If you have sideslip or lateral velocity, include it. The single-track (bicycle) kinematic/dynamic model maps control inputs (wheel angle, throttle/brake) and process noise to state evolution — many references and a numerically stable discrete form exist. 
arXiv
+1

Bicycle model (kinematic form, discrete)

A standard discrete kinematic bicycle (time step 
Δt
Δt):

xk+1	=xk+vkcos⁡(ψk)Δt
yk+1	=yk+vksin⁡(ψk)Δt
ψk+1	=ψk+vkLtan⁡(δk)Δt
vk+1	=vk+akΔt
x
k+1
	​

y
k+1
	​

ψ
k+1
	​

v
k+1
	​

	​

=x
k
	​

+v
k
	​

cos(ψ
k
	​

)Δt
=y
k
	​

+v
k
	​

sin(ψ
k
	​

)Δt
=ψ
k
	​

+
L
v
k
	​

	​

tan(δ
k
	​

)Δt
=v
k
	​

+a
k
	​

Δt
	​


(Use the dynamic single-track if lateral tire dynamics and low-level control are important.) 
MathWorks
+1

UKF specifics

State vector 
x
x, covariance 
P
P.

Process noise covariance 
Q
Q tuned from sensor/vehicle dynamics; measurement noise 
R
R from sensors. Consider adaptive tuning if conditions change. 
MDPI
+1

Use the Unscented Transform (sigma points) to propagate nonlinearity exactly to 2-3rd order (UKF advantage over EKF for highly nonlinear bicycle model).

Implement square-root or numerically stable UKF variant (Cholesky) to maintain positive definiteness.

Reference implementations / algorithmic details: standard UKF descriptions and vehicle-estimation papers. 
PMC
+1

3) Predict to discrete future times and compute uncertainty ellipse

At each decision time 
t0
t
0
	​

 you want probabilities at a (set of) future times 
tf=t0+τ
t
f
	​

=t
0
	​

+τ (e.g., every 0.1 s out to 5 s):

Use your UKF process model to predict the mean state 
xˉ(tf)
x
ˉ
(t
f
	​

) and covariance 
P(tf)
P(t
f
	​

) for each vehicle (the UKF naturally provides this).

Compute the relative state between two vehicles:

Δx=xego−xother
Δx=x
ego
	​

−x
other
	​


and the combined covariance

PΔ=Pego+Pother
P
Δ
	​

=P
ego
	​

+P
other
	​


(neglecting cross-correlations unless they exist; if sensors share measurements, include cross terms). NASA/space literature uses the same “combined covariance” idea for Pc. 
NASA Technical Reports Server
+1

Project the relative position covariance into the plane of interest (see next section) to form the 2×2 covariance used for the miss-ellipse.

Often you’ll draw the uncertainty ellipse from the 2×2 position covariance: the eigenvectors/eigenvalues of that 2×2 give principal axes and lengths (e.g. 1-sigma ellipse = 
λi
λ
i
	​
