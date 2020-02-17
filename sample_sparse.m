# preconditioning
# instead of solving Ax = b,
# we solve P^-1 A x = P^-1 b
# Let's declare M = P^-1
# incomplete LU (ilu)
## L or U might be more denser than A, causing memory issue (or storage)
## So iLU is recommended - but inverse of iLU would be denser - how to resolve?
# polynomial preconditioner
# AMG - not in octave yet

N = 50 # if N is too small like 10 or 20, we may not see the effect of preconditioning
Nnear = N*0.7
A = zeros(N,N);
b = zeros(N,1);
% well-defined-symmetric
for i=1:N
  for j=1:N
    tmp = N - 2*abs(i-j);
    if (tmp > Nnear)
      A(i,j) = rand(1);      
      if (i==j)
        A(i,j) = A(i,j) + 0.; # diagonal terms are intentionally bigger
      endif
    endif
   endfor  
   b(i) = 1;
endfor
x = inv(A)*b;
# 1. brute force bicgstab
# As A is not symmetric, pcg cannot be used
x_bf = bicgstab(A,b,1e-6,10000); # took 293 steps or convergence failed when N=50
# 2. Jacobi preconditioner = diagonal components only. The simplest
P=zeros(N,N);
for i=1:N
  P(i,i) = A(i,i);
endfor
M = inv(P);
x_j  = bicgstab(M*A,M*b,1e-6,10000); # took 256 steps or 346 steps for N=50
# As N increases like 100, Jacobi preconditioner may not work anymore
# 3. ILU 
[L,U,P]= ilu(sparse(A));
# A -> sparse matrix -> incomplete LU -> full matrix again
M = inv(full(L));
x  = bicgstab(M*A,M*b,1e-6,10000); # took 217 steps or 213 steps for N=50
