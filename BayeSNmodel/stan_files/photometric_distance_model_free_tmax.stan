functions {
	matrix spline_coeffs_irr(vector x, vector xk, matrix KD, real ext) {
		int Nx = rows(x);
		int Nk = rows(xk);
		matrix[Nx,Nk] X;

		if (max(x) > max(xk)+ext || min(x) < min(xk)-ext) {
			reject("Spline interpolation error:", x);
		}

		X = rep_matrix(0,Nx,Nk);

		for (n in 1:Nx) {
			real h;
			real a;
			real b;
			if (x[n] > xk[Nk]) {
				real f;

				h = xk[Nk] - xk[Nk-1];
				a = (xk[Nk] - x[n])/h;
				b = 1 - a;
				f = (x[n] - xk[Nk])*h/6;

				X[n,Nk-1] = a;
				X[n,Nk] = b;
				X[n,:] += f*KD[Nk-1,:];
			} else if (x[n] < xk[1]) {
				real f;

				h = xk[2] - xk[1];
				b = (x[n] - xk[1])/h;
				a = 1 - b;
				f = (x[n] - xk[1])*h/6;

				X[n,1] = a;
				X[n,2] = b;
				X[n,:] -= f*KD[2,:];
			} else {
				int q;
				real c;
				real d;

				for (k in 1:Nk-1) {
					if (xk[Nk-k] <= x[n]) {
						q = Nk-k;
						break;
					}
				}

				h = xk[q+1] - xk[q];
				a = (xk[q+1] -x[n])/h;
				b = 1 - a;
				c = ((pow(a,3) - a)/6.0)*square(h);
				d = ((pow(b,3) - b)/6.0)*square(h);

				X[n,q] = a;
				X[n,q+1] = b;
				X[n,:] += c*KD[q,:] + d*KD[q+1,:];
			}
		}

		return X;
	}
	vector interpolate_hsiao(real t, vector th, matrix fh) {
		int Nh = rows(th);

		int q0;
		real t0;
		real t1;
		vector[cols(fh)] f0;
		vector[cols(fh)] f1;
		vector[cols(fh)] f;
	
		for (n in 1:Nh) {
			if (t < th[n]) {
				q0 = n-1;
				t0 = th[n-1];
				t1 = th[n];
				break;
			}
		}

		f0 = to_vector(fh[q0,:]);
		f1 = to_vector(fh[q0+1,:]);

		f = (f0*(t1-t) + f1*(t-t0))/(t1-t0);

		return f;
	}
}
data {
	//Total number of observations
	int<lower=1> Nobs;
	//Total number of interpolation points
	int<lower=1> Dint;
	
	//Number of knots
	int<lower=1> Nknots;
	int<lower=1> Ntknots;
	int<lower=1> Nlknots;

	//Knot values
	vector[Ntknots] tk_unique;
	real tk[Nknots];
	real lk[Nknots];

	//Supernova data
	vector[Nobs] tobs;
	vector[Nobs] fobs;
	vector[Nobs] fobserr;

	//Heliocentric redshift
	real zhel;

	//Guess for peak time (prior is this +/- tmax_shift days)
	real tmax_guess;
	//Permitted shift in phase of peak.
	real phase_shift;

	//Precomputed vectors
	//A_lambda/A_V
	vector[Dint] Alam_AV;

	//CSR format precomputed matrices
	//Wavelength interpolation matrix
	int Jl_nnz;
	vector[Jl_nnz] Jl_w;
	int Jl_v[Jl_nnz];
	int Jl_u[Dint+1];
	//Passband integral
	int H_nnz;
	vector[H_nnz] H_w;
	int H_v[H_nnz];
	int H_u[Nobs+1];
	
	//Part of the spline matrix
	matrix[Ntknots,Ntknots] KDinv;
		
	//Stack of Hsiao template values to interpolate
	int Nhsiao;
	int nint[Nobs+1];
	vector[Nhsiao] thsiao;
	matrix[Nhsiao,Dint] fhsiao;

	//Distance prior params
	real muhat;
	real muhaterr;

	//SNANA zero_point
	real ZPT;

	//SED model parameters
	matrix[Nlknots,Ntknots] W0;
	matrix[Nlknots,Ntknots] W1;
	matrix[Nknots-2*Ntknots,Nknots-2*Ntknots] L_Sigma;
	real<lower=0> tauA;
	real M0;
	real<lower=0> sigma0;
}
transformed data {
	//Convenient factor
	real gamma = log(10)/2.5;
	//Permitted shift in observer fram time of max
	real tmax_shift;

	//Number of unpinned epsilons
	int Nfree;

	//Standard deviation of the del_M + mu sampling variable
	real Dserr;

	tmax_shift = phase_shift*(1.0 + zhel);
	
	Dserr = sqrt(square(muhaterr) + square(sigma0));
			
	Nfree = Nknots - 2*Ntknots;
}
parameters {
	real<lower=-tmax_shift,upper=tmax_shift> dtmax;
	real theta;
	real Ds;
	real<lower=0> AV;
	vector[Nfree] epsilon_tform; //Non-centred transform of epsilon matrix
}
transformed parameters {
	real tmax;
	vector[Nfree] epsilon_free;
	matrix[Nlknots,Ntknots] epsilon;
	vector[Nobs] phase;
	
	tmax = tmax_guess + dtmax;

	epsilon_free = L_Sigma*epsilon_tform;
	epsilon = rep_matrix(0, Nlknots, Ntknots);
	epsilon[2:Nlknots-1, :] = to_matrix(epsilon_free, Nlknots-2, Ntknots);

	phase = (tobs - tmax)/(1.0+zhel);
}
model {
	//Declare a variable for the latent fluxes/mags
	vector[Nobs] f;

	//Hsiao template
	vector[Dint] S0;
	//Time interpolation matrix
	matrix [Ntknots,Nobs] Jt;

	Jt = spline_coeffs_irr(phase, tk_unique, KDinv, phase_shift)';
	for (n in 1:Nobs) {
		S0[nint[n]:nint[n+1]-1] = interpolate_hsiao(phase[n], thsiao, fhsiao[:,nint[n]:nint[n+1]-1]);
	}
	
	//Sampling statements for parameters
	dtmax ~ uniform(-tmax_shift, tmax_shift);
	theta ~ std_normal();
	Ds ~ normal(muhat, Dserr);
	AV ~ exponential(inv(tauA));
	epsilon_tform ~ std_normal();
	
	//Calculate latent flux in a scope
	{
		vector[Nlknots*Nobs] WJt;
		vector[Dint] JlWJt;
		vector[Dint] Alam; //Host extinction at obs wavelengths
		vector[Dint] Stilde; //Host-extinguised SED
		vector[Nobs] HS;

		matrix[Nlknots,Ntknots] W; //Knots defining intrinsic SED

		//The sum of the intrinsic FCs and epsilon matrix
		W = W0 + theta*W1 + epsilon;
		
		//A matrix containing W*Jt in each column (reshaped into a vector)
		WJt = to_vector(W*Jt);
		
		//Interpolate intrinsic SED to obs times and wavelengths
		JlWJt = csr_matrix_times_vector(Dint, Nlknots*Nobs, Jl_w, Jl_v, Jl_u, WJt);
		Alam = Alam_AV*AV;
		
		//Multiply by Hsiao template	
		Stilde = S0 .* exp(-gamma*(JlWJt + Alam));
		//Integrate through passbands
		HS = csr_matrix_times_vector(Nobs, Dint, H_w, H_v, H_u, Stilde);
		//Correct for distance and mean abs magnitude
		f = exp(gamma*(ZPT - M0 - Ds)) * HS;
	}

	//Data likelihood
	fobs ~ normal(f, fobserr);
}
generated quantities {
	real delM;
	real mu;

	//Extract mu and del_M from their linear combination
	mu = normal_rng((Ds*square(muhaterr) + muhat*square(sigma0))/square(Dserr), sqrt((square(sigma0)*square(muhaterr))/square(Dserr)));
	delM = Ds - mu;
}
