!-----------------------------------------------------------------------------!
! 2D Euler Solver with WENO Scheme
!-----------------------------------------------------------------------------!
! Omer San, December 2012, Virginia Tech, omersan@vt.edu
!-----------------------------------------------------------------------------!

program Euler2D
implicit none
integer:: nx,ny,ip,nt,k
real*8 :: cfl,dt,dx,dy,lx,ly,time,tmax,t1,t2
real*8,allocatable:: q(:,:,:)


!problem selection:
ip=6 ![1]c3,[2]c5,[3]c16,[4]c4,[5]c6,[6]c12,[7]c15,[8]c17,[9]c8,[10]c11,[11]c2
!differnet Rieamann configurations.

!inputs
nx = 400
ny = nx
lx = 1.0d0
ly = lx
cfl = 0.5d0 ! we use to compute time step
!the code was orginally using adaptive time step, but I make it constant time step to make data analysis simpler.
!so we are not using this cfl within time iterations, I am just using at the beginning to get a rough estimate about dt.


!all problems are defined in interval of [0,lx]x[0,ly]
!spatial step size
dx = lx/dfloat(nx)
dy = ly/dfloat(ny)

allocate(q(-2:nx+3,-2:ny+3,4)) !field


!initial conditions and problem definition
call problem(nx,ny,dx,dy,tmax,ip,q)

!compute time step from cfl
call timestep(nx,ny,dx,dy,cfl,dt,q)

!maximum number of time steps
nt = nint(tmax/dt)
nt = nt-mod(nt,100)+100

!fixed time step
dt = tmax/dfloat(nt)


call cpu_time(t1)

time = 0.0d0

!Time integration
do k=1,nt
  time = time + dt
  call tvdrk3(nx,ny,dx,dy,dt,q)
  print*,k,time
  if (mod(k,2)==0) then
    call outresult(nx,ny,dx,dy,time,q,k/2)
  end if
end do

call cpu_time(t2)

!writing results

!write data
! open(19,file='cpu.txt')
! write(19,*)"cpu time =", t2-t1, "  second"
! write(19,*)"cpu time =", (t2-t1)/60.0d0, "  minute"
! write(19,*)"cpu time =", (t2-t1)/60.0d0/60.0d0, "  hour"
! close(19)


end

!-----------------------------------------------------------------------------------!
!TVD Runge Kutta 3rd order
!-----------------------------------------------------------------------------------!
subroutine tvdrk3(nx,ny,dx,dy,dt,q)
implicit none
integer::nx,ny,i,j,m
real*8 ::dx,dy,dt,a,b
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: u(:,:,:),s(:,:,:)


allocate(u(-2:nx+3,-2:ny+3,4))
allocate(s(nx,ny,4))

a = 1.0d0/3.0d0
b = 2.0d0/3.0d0


call rhs(nx,ny,dx,dy,q,s)


  do m=1,4
    do j=1,ny
      do i=1,nx
        u(i,j,m) = q(i,j,m) + dt*s(i,j,m)
      end do
    end do
  end do


call rhs(nx,ny,dx,dy,u,s)

  do m=1,4
    do j=1,ny
      do i=1,nx
      u(i,j,m) = 0.75d0*q(i,j,m) + 0.25d0*u(i,j,m) + 0.25d0*dt*s(i,j,m)
      end do
    end do
  end do

call rhs(nx,ny,dx,dy,u,s)

  do m=1,4
    do j=1,ny
      do i=1,nx
        q(i,j,m) = a*q(i,j,m)+b*u(i,j,m)+b*dt*s(i,j,m)
      end do
    end do
  end do

deallocate (u,s)

return
end


!-----------------------------------------------------------------------------------!
!Computing results
!Compute primitive variables from conservative variables
!-----------------------------------------------------------------------------------!
! subroutine write_2d(nx,ny,field,field_name)
!   open(0,file=field_name)
!   do j = 1,ny
!     do i = 1,nx-1
!       write(0,'(1x,F8.4,A)',advance='no') field(i,j),","
!     end do
!     write(0,'(1x,F8.4)') field(nx,j)
!   end do
!   close(0)
! end subroutine

subroutine outresult(nx,ny,dx,dy,time,q,n_plot)
implicit none
integer::nx,ny,i,j,n_plot
real*8 ::dx,dy,time,gamma
real*8 ::q(-2:nx+3,-2:ny+3,4)
real*8,allocatable:: x(:),y(:),r(:,:),u(:,:),v(:,:),p(:,:),e(:,:),h(:,:),a(:,:),m(:,:)
character (len=20) :: folder
character (len=20) :: suffix

common /fluids/ gamma

allocate(x(nx),y(ny),r(nx,ny),u(nx,ny),v(nx,ny))
allocate(p(nx,ny),e(nx,ny),h(nx,ny),a(nx,ny),m(nx,ny))

!computing results at cell centers:
do j=1,ny
  do i=1,nx
    x(i) = dfloat(i)*dx - 0.5d0*dx
    y(j) = dfloat(j)*dy - 0.5d0*dy

    r(i,j)= q(i,j,1)
    u(i,j)= q(i,j,2)/q(i,j,1)
    v(i,j)= q(i,j,3)/q(i,j,1)
    e(i,j)= q(i,j,4)/q(i,j,1)
    p(i,j)= (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
    h(i,j)= e(i,j) + p(i,j)/r(i,j)
    a(i,j)= dsqrt(gamma*p(i,j)/r(i,j))
    m(i,j)= dsqrt(u(i,j)*u(i,j)+v(i,j)*v(i,j))/a(i,j)
  end do
end do

folder = "Results/Euler_3/"
if (n_plot < 10) then
  write(suffix, "(I1,A4)") n_plot,".csv"
else if (n_plot < 100) then
  write(suffix, "(I2,A4)") n_plot,".csv"
else
  write(suffix, "(I3,A4)") n_plot,".csv"
end if

!writing results at cell centers:
open(0,file=trim(folder)//'X'//trim(suffix))
do j=1,ny
  do i=1,nx-1
    write(0,'(1x,F8.4,A)',advance='no') x(i),","
  end do
  write(0,'(1x,F8.4)') x(nx)
end do
close(0)

open(0,file=trim(folder)//'Y'//trim(suffix))
do j=1,ny
  do i=1,nx-1
    write(0,'(1x,F8.4,A)',advance='no') y(j),","
  end do
  write(0,'(1x,F8.4)') y(j)
end do

close(0)

open(0,file=trim(folder)//"U"//trim(suffix))
do j = 1,ny
  do i = 1,nx-1
    write(0,'(1x,F8.4,A)',advance='no') u(i,j),","
  end do
  write(0,'(1x,F8.4)') u(nx,j)
end do
close(0)

open(0,file=trim(folder)//"V"//trim(suffix))
do j = 1,ny
  do i = 1,nx-1
    write(0,'(1x,F8.4,A)',advance='no') v(i,j),","
  end do
  write(0,'(1x,F8.4)') v(nx,j)
end do
close(0)

open(0,file=trim(folder)//"P"//trim(suffix))
do j = 1,ny
  do i = 1,nx-1
    write(0,'(1x,F8.4,A)',advance='no') p(i,j),","
  end do
  write(0,'(1x,F8.4)') p(nx,j)
end do
close(0)

!writing results at cell centers:
! open(20,file='field_all.plt')
! write(20,*) 'variables ="x","y","r","u","v","p","e","h","a","m"' ! write(20,*) 'zone t ="time ',time,'", i=',nx,',j=',ny,', f = point'
! do j=1,ny
! do i=1,nx
! write(20,*)x(i),y(j),r(i,j),u(i,j),v(i,j),p(i,j),e(i,j),h(i,j),a(i,j),m(i,j)
! end do
! end do
! close(20)
!
! !writing results at cell centers:
! open(30,file='field_density.plt')
! write(30,*) 'variables ="x","y","r"'
! write(30,*) 'zone t ="time ',time,'", i=',nx,',j=',ny,', f = point'
! do j=1,ny
! do i=1,nx
! write(30,*)x(i),y(j),r(i,j)
! end do
! end do
! close(30)
!
! !writing results at cell centers:
! open(40,file='field_mach.plt')
! write(40,*) 'variables ="x","y","m"'
! write(40,*) 'zone t ="time ',time,'", i=',nx,',j=',ny,', f = point'
! do j=1,ny
! do i=1,nx
! write(40,*)x(i),y(j),m(i,j)
! end do
! end do
! close(40)
!
! !writing results at cell centers:
! open(50,file='field_pressure.plt')
! write(50,*) 'variables ="x","y","p"'
! write(50,*) 'zone t ="time ',time,'", i=',nx,',j=',ny,', f = point'
! do j=1,ny
! do i=1,nx
! write(50,*)x(i),y(j),p(i,j)
! end do
! end do
! close(50)

deallocate(x,y,r,u,v,p,e,h,a,m)

return
end


!-----------------------------------------------------------------------------------!
!Initial Conditions and Problem Definition
!SIAM J. SCI. COMPUT.
!Society for Industrial and Applied Mathematics
!Vol. 19, No. 2, pp. 319{340, March 1998
!SOLUTION OF TWO-DIMENSIONAL RIEMANN PROBLEMS OF GAS DYNAMICS BY POSITIVE SCHEMES
!PETER D. LAX AND XU-DONG LIU
!-----------------------------------------------------------------------------------!
subroutine problem(nx,ny,dx,dy,tmax,ip,q)
implicit none
integer::nx,ny,ip,i,j
real*8 ::r,u,v,p,e,x,y,dx,dy,x0,y0
real*8 ::rUL,rUR,rLL,rLR,uUL,uUR,uLL,uLR,vUL,vUR,vLL,vLR,pUL,pUR,pLL,pLR
real*8 ::gamma,tmax
real*8 ::q(-2:nx+3,-2:ny+3,4)

common /fluids/ gamma


if (ip.eq.1) then !Lax-Lio Configuration 3

  	!upper-left
    rUL=0.5323d0
	uUL=1.206d0
    vUL=0.0d0
    pUL=0.3d0
	!upper-right
    rUR=1.5d0
	uUR=0.0d0
    vUR=0.0d0
    pUR=1.5d0
	!lower-left
    rLL=0.138d0
	uLL=1.206d0
    vLL=1.206d0
    pLL=0.029d0
	!lower-right
    rLR=0.5323d0
	uLR=0.0d0
    vLR=1.206d0
    pLR=0.3d0

    gamma = 1.4d0
    tmax  = 0.3d0


else if (ip.eq.2) then !Lax-Liu Configuration 5

  	!upper-left
    rUL=2.0d0
	uUL=-0.75d0
    vUL=0.5d0
    pUL=1.0d0
	!upper-right
    rUR=1.0d0
	uUR=-0.75d0
    vUR=-0.5d0
    pUR=1.0d0
	!lower-left
    rLL=1.0d0
	uLL=0.75d0
    vLL=0.5d0
    pLL=1.0d0
	!lower-right
    rLR=3.0d0
	uLR=0.75d0
    vLR=-0.5d0
    pLR=1.0d0


    gamma = 1.4d0
    tmax  = 0.23d0


else if (ip.eq.3) then !Lax-Liu Configuration 16

  	!upper-left
    rUL=1.0222d0
	uUL=-0.6179d0
    vUL=0.1d0
    pUL=1.0d0
	!upper-right
    rUR=0.5313d0
	uUR=-0.1d0
    vUR=0.1d0
    pUR=0.4d0
	!lower-left
    rLL=0.8d0
	uLL=0.1d0
    vLL=0.1d0
    pLL=1.0d0
	!lower-right
    rLR=1.0d0
	uLR=0.1d0
    vLR=0.8276d0
    pLR=1.0d0


    gamma = 1.4d0
    tmax  = 0.2d0

else if (ip.eq.4) then !Lax-Liu Configuration 4

  	!upper-left
    rUL=0.5065d0
	uUL=0.8939d0
    vUL=0.0d0
    pUL=0.35d0
	!upper-right
    rUR=1.1d0
	uUR=0.0d0
    vUR=0.0d0
    pUR=1.1d0
	!lower-left
    rLL=1.1d0
	uLL=0.8939d0
    vLL=0.8939d0
    pLL=1.1d0
	!lower-right
    rLR=0.5065d0
	uLR=0.0d0
    vLR=0.8939d0
    pLR=0.35d0


    gamma = 1.4d0
    tmax  = 0.25d0


else if (ip.eq.5) then !Lax-Liu Configuration 6

  	!upper-left
    rUL=2.0d0
	uUL=0.75d0
    vUL=0.5d0
    pUL=1.0d0
	!upper-right
    rUR=1.0d0
	uUR=0.75d0
    vUR=-0.5d0
    pUR=1.0d0
	!lower-left
    rLL=1.0d0
	uLL=-0.75d0
    vLL=0.5d0
    pLL=1.0d0
	!lower-right
    rLR=3.0d0
	uLR=-0.75d0
    vLR=-0.5d0
    pLR=1.0d0


    gamma = 1.4d0
    tmax  = 0.25d0

else if (ip.eq.6) then !Lax-Liu Configuration 12

  	!upper-left
    rUL=1.0d0
	uUL=0.7276d0
    vUL=0.0d0
    pUL=1.0d0
	!upper-right
    rUR=0.5313d0
	uUR=0.0d0
    vUR=0.0d0
    pUR=0.3d0
	!lower-left
    rLL=0.8d0
	uLL=0.0d0
    vLL=0.0d0
    pLL=1.0d0
	!lower-right
    rLR=1.0d0
	uLR=0.0d0
    vLR=0.7276d0
    pLR=1.0d0


    gamma = 1.4d0
    tmax  = 0.25d0

else if (ip.eq.7) then !Lax-Liu Configuration 15

  	!upper-left
    rUL=0.5197d0
	uUL=-0.6259d0
    vUL=-0.3d0
    pUL=0.4d0
	!upper-right
    rUR=1.0d0
	uUR=0.1d0
    vUR=-0.3d0
    pUR=1.0d0
	!lower-left
    rLL=0.8d0
	uLL=0.1d0
    vLL=-0.3d0
    pLL=0.4d0
	!lower-right
    rLR=0.5313d0
	uLR=0.1d0
    vLR=0.4276d0
    pLR=0.4d0


    gamma = 1.4d0
    tmax  = 0.2d0

else if (ip.eq.8) then !Lax-Liu Configuration 17

  	!upper-left
    rUL=2.0d0
	uUL=0.0d0
    vUL=-0.3d0
    pUL=1.0d0
	!upper-right
    rUR=1.0d0
	uUR=0.0d0
    vUR=-0.4d0
    pUR=1.0d0
	!lower-left
    rLL=1.0625d0
	uLL=0.0d0
    vLL=0.2145d0
    pLL=0.4d0
	!lower-right
    rLR=0.5197d0
	uLR=0.0d0
    vLR=-1.1259d0
    pLR=0.4d0


    gamma = 1.4d0
    tmax  = 0.3d0

else if (ip.eq.9) then !Lax-Liu Configuration 8

  	!upper-left
    rUL=1.0d0
	uUL=-0.6259d0
    vUL=0.1d0
    pUL=1.0d0
	!upper-right
    rUR=0.5197d0
	uUR=0.1d0
    vUR=0.1d0
    pUR=0.4d0
	!lower-left
    rLL=0.8d0
	uLL=0.1d0
    vLL=0.1d0
    pLL=1.0d0
	!lower-right
    rLR=1.0d0
	uLR=0.1d0
    vLR=-0.6259d0
    pLR=1.0d0


    gamma = 1.4d0
    tmax  = 0.25d0

else if (ip.eq.10) then !Lax-Liu Configuration 11

  	!upper-left
    rUL=0.5313d0
	uUL=0.8276d0
    vUL=0.0d0
    pUL=0.4d0
	!upper-right
    rUR=1.0d0
	uUR=0.1d0
    vUR=0.0d0
    pUR=1.0d0
	!lower-left
    rLL=0.8d0
	uLL=0.1d0
    vLL=0.0d0
    pLL=0.4d0
	!lower-right
    rLR=0.5313d0
	uLR=0.1d0
    vLR=0.7276d0
    pLR=0.4d0


    gamma = 1.4d0
    tmax  = 0.3d0

else !Lax-Liu Configuration 2

  	!upper-left
    rUL=0.5197d0
	uUL=-0.7259d0
    vUL=0.0d0
    pUL=0.4d0
	!upper-right
    rUR=1.0d0
	uUR=0.0d0
    vUR=0.0d0
    pUR=1.0d0
	!lower-left
    rLL=1.0d0
	uLL=-0.7259d0
    vLL=-0.7259d0
    pLL=1.0d0
	!lower-right
    rLR=0.5197d0
	uLR=0.0d0
    vLR=-0.7259d0
    pLR=0.4d0


    gamma = 1.4d0
    tmax  = 0.2d0

end if

!---------------------------------------------------------!
!construction initial conditions for conserved variables
!---------------------------------------------------------!
x0 = 0.5d0
y0 = 0.5d0

do j=-2,ny+3
do i=-2,nx+3

	x = dfloat(i)*dx - 0.5d0*dx !cell-centered definition
    y = dfloat(j)*dy - 0.5d0*dy !cell-centered definition

  	if(x.le.x0.and.y.gt.y0) then !upper-left


    	r = rUL
		u = uUL
    	v = vUL
    	p = pUL

	else if(x.gt.x0.and.y.gt.y0) then !upper-right


       	r = rUR
		u = uUR
    	v = vUR
    	p = pUR

	else if(x.gt.x0.and.y.le.y0) then !lower-right

		r = rLR
		u = uLR
    	v = vLR
    	p = pLR

    else !lower-left

        r = rLL
		u = uLL
    	v = vLL
    	p = pLL


    end if

		e=p/(r*(gamma-1.0d0))+0.5d0*(u*u+v*v)


    !construct conservative variables

	q(i,j,1)=r
	q(i,j,2)=r*u
    q(i,j,3)=r*v
	q(i,j,4)=r*e

end do
end do


return
end



!-----------------------------------------------------------------------------------!
!Computing Right Hand Side
!-----------------------------------------------------------------------------------!
subroutine rhs(nx,ny,dx,dy,q,s)
implicit none
integer::nx,ny,i,j,m
real*8 ::dx,dy,gamma,l1,l2,l3,rad0,rad1,ps,p,a
real*8 ::q(-2:nx+3,-2:ny+3,4),s(nx,ny,4)
real*8,allocatable:: u(:,:),uL(:,:),uR(:,:),hL(:,:),hR(:,:),f(:,:,:),g(:,:,:)

common /fluids/ gamma

!------------------------------------------!
!Transmissive boundary conditions (open bc)
!------------------------------------------!
do m=1,4

	do i=1,nx
	q(i, 0,m)  =q(i,1,m) 	!bottom of the domain
    q(i,-1,m)  =q(i,2,m)	!bottom of the domain
    q(i,-2,m)  =q(i,3,m)	!bottom of the domain

	q(i,ny+1,m)=q(i,ny,m) 	!top of the domain
	q(i,ny+2,m)=q(i,ny-1,m) !top of the domain
	q(i,ny+3,m)=q(i,ny-2,m) !top of the domain
	end do

	do j=-2,ny+3
	q( 0,j,m)  =q(1,j,m)	!left of the domain
    q(-1,j,m)  =q(2,j,m)	!left of the domain
    q(-2,j,m)  =q(3,j,m)	!left of the domain

	q(nx+1,j,m)=q(nx,j,m) 	!right of the domain
	q(nx+2,j,m)=q(nx-1,j,m)	!right of the domain
	q(nx+3,j,m)=q(nx-2,j,m)	!right of the domain
	end do

end do


!-------------------------------------------------!
!Riemann: Construction based schemes
!-------------------------------------------------!

allocate(f(0:nx,ny,4),g(nx,0:ny,4))

!-----------------------------!
!Compute x-fluxes for all j
!-----------------------------!
allocate(u(-2:nx+3,4),uL(0:nx,4),uR(0:nx,4),hL(0:nx,4),hR(0:nx,4))

do j=1,ny

	!assign q vector as u in x-direction
    do m=1,4
    do i=-2,nx+3
    u(i,m)=q(i,j,m)
    end do
    end do

	!Reconstruction scheme
	!WENO5 construction
	!call weno5_old(nx,u,uL,uR)
    call weno5L(nx,u,uL)
    call weno5R(nx,u,uR)


	!compute left and right fluxes
  	call xflux(nx,uL,hL)
	call xflux(nx,uR,hR)

    !-----------------------------!
	!Rusanov flux in x-direction
    !-----------------------------!

	do i=0,nx

	!at point i
    p = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
	a = dsqrt(gamma*p/q(i,j,1))

	l1=dabs(q(i,j,2)/q(i,j,1))
	l2=dabs(q(i,j,2)/q(i,j,1) + a)
	l3=dabs(q(i,j,2)/q(i,j,1) - a)
	rad0 = max(l1,l2,l3)

    !at point i+1
    p = (gamma-1.0d0)*(q(i+1,j,4)-0.5d0*(q(i+1,j,2)*q(i+1,j,2)/q(i+1,j,1)+q(i+1,j,3)*q(i+1,j,3)/q(i+1,j,1)))
	a = dsqrt(gamma*p/q(i+1,j,1))

	l1=dabs(q(i+1,j,2)/q(i+1,j,1))
	l2=dabs(q(i+1,j,2)/q(i+1,j,1) + a)
	l3=dabs(q(i+1,j,2)/q(i+1,j,1) - a)
	rad1 = max(l1,l2,l3)

    ps = max(rad0,rad1)

		!compute flux in x-direction
		do m=1,4
    	f(i,j,m)=0.5d0*((hR(i,m)+hL(i,m)) - ps*(uR(i,m)-uL(i,m)))
		end do

    end do

end do
deallocate(u,uL,uR,hL,hR)

!-----------------------------!
!Compute y-fluxes for all i
!-----------------------------!
allocate(u(-2:ny+3,4),uL(0:ny,4),uR(0:ny,4),hL(0:ny,4),hR(0:ny,4))

do i=1,nx

	!assign q vector as u in y-direction
    do m=1,4
    do j=-2,ny+3
    u(j,m)=q(i,j,m)
    end do
    end do

	!Reconstruction scheme
	!WENO5 construction
	!call weno5_old(ny,u,uL,uR)
	call weno5L(ny,u,uL)
    call weno5R(ny,u,uR)

	!compute left and right fluxes
  	call yflux(ny,uL,hL)
	call yflux(ny,uR,hR)

	!characteristic speed for Rusanov flux


	do j=0,ny

  	!at point j
    p = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
	a = dsqrt(gamma*p/q(i,j,1))

	l1=dabs(q(i,j,3)/q(i,j,1))
	l2=dabs(q(i,j,3)/q(i,j,1) + a)
	l3=dabs(q(i,j,3)/q(i,j,1) - a)
	rad0 = max(l1,l2,l3)

	!at point j+1
    p = (gamma-1.0d0)*(q(i,j+1,4)-0.5d0*(q(i,j+1,2)*q(i,j+1,2)/q(i,j+1,1)+q(i,j+1,3)*q(i,j+1,3)/q(i,j+1,1)))
	a = dsqrt(gamma*p/q(i,j+1,1))

	l1=dabs(q(i,j+1,3)/q(i,j+1,1))
	l2=dabs(q(i,j+1,3)/q(i,j+1,1) + a)
	l3=dabs(q(i,j+1,3)/q(i,j+1,1) - a)
	rad1 = max(l1,l2,l3)

    ps = max(rad0,rad1)

		do m=1,4
    	!compute flux in y-direction
    	g(i,j,m)=0.5d0*((hR(j,m)+hL(j,m)) - ps*(uR(j,m)-uL(j,m)))
		end do

    end do

end do
deallocate(u,uL,uR,hL,hR)

!--------------!
!Compute rhs
!--------------!
do m=1,4
do j=1,ny
do i=1,nx
	s(i,j,m)=-(f(i,j,m)-f(i-1,j,m))/dx - (g(i,j,m)-g(i,j-1,m))/dy
end do
end do
end do

deallocate(f,g)



end


!-----------------------------------------------------------------------------!
!WENO5 reconstruction for upwind direction (positive & left to right)
!-----------------------------------------------------------------------------!
subroutine weno5L(n,u,f)
implicit none
integer::n
real*8 ::u(-2:n+3,4),f(0:n,4)
integer::i,m
real*8 ::a,b,c,d,e,w

do m=1,4
do i=0,n
  	a = u(i-2,m)
  	b = u(i-1,m)
  	c = u(i,m)
  	d = u(i+1,m)
  	e = u(i+2,m)
  	call weno5(a,b,c,d,e,w)
  	f(i,m) = w  !gives right value at i+1/2
end do
end do

return
end

!-----------------------------------------------------------------------------!
!WENO5 reconstruction for downwind direction (negative & right to left)
!-----------------------------------------------------------------------------!
subroutine weno5R(n,u,f)
implicit none
integer::n
real*8 ::u(-2:n+3,4),f(0:n,4)
integer::i,m
real*8 ::a,b,c,d,e,w

do m=1,4
do i=1,n+1
  	a = u(i+2,m)
  	b = u(i+1,m)
  	c = u(i,m)
  	d = u(i-1,m)
  	e = u(i-2,m)
  	call weno5(a,b,c,d,e,w)
  	f(i-1,m) = w  !gives left value at i-1/2
end do
end do

return
end

!----------------------------------------------------------------------------------!
!WENO5
!----------------------------------------------------------------------------------!
subroutine weno5(a,b,c,d,e,f)
implicit none
real*8 ::a,b,c,d,e,f
real*8 ::q1,q2,q3
real*8 ::s1,s2,s3
real*8 ::a1,a2,a3
real*8 ::eps

q1 = a/3.0d0 - 7.0d0/6.0d0*b + 11.0d0/6.0d0*c
q2 =-b/6.0d0 + 5.0d0/6.0d0*c + d/3.0d0
q3 = c/3.0d0 + 5.0d0/6.0d0*d - e/6.0d0

s1 = 13.0d0/12.0d0*(a-2.0d0*b+c)**2 + 0.25d0*(a-4.0d0*b+3.0d0*c)**2
s2 = 13.0d0/12.0d0*(b-2.0d0*c+d)**2 + 0.25d0*(d-b)**2
s3 = 13.0d0/12.0d0*(c-2.0d0*d+e)**2 + 0.25d0*(3.0d0*c-4.0d0*d+e)**2

!Jiang-Shu estimator
eps = 1.0d-6
a1 = 1.0d-1/(eps+s1)**2
a2 = 6.0d-1/(eps+s2)**2
a3 = 3.0d-1/(eps+s3)**2

!Shen-Zha estimator
!eps = 1.0d-20
!a1 = 1.0d-1*(1.0d0 + (dabs(s1-s3)/(eps+s1))**2)
!a2 = 6.0d-1*(1.0d0 + (dabs(s1-s3)/(eps+s2))**2)
!a3 = 3.0d-1*(1.0d0 + (dabs(s1-s3)/(eps+s3))**2)

f = (a1*q1 + a2*q2 + a3*q3)/(a1 + a2 + a3)

return
end subroutine




!-----------------------------------------------------------------------------------!
!5th order WENO construction
!-----------------------------------------------------------------------------------!
subroutine weno5_old(n,q,qL,qR)
implicit none
integer::n,m,i
real*8 ::q(-2:n+3,4),qL(0:n,4),qR(0:n,4)
real*8 ::eps,a,b,c,h,g,a0,a1,a2,w0,w1,w2,b0,b1,b2

	eps = 1.0d0-6

    h = 13.0d0/12.0d0
    g = 1.0d0/6.0d0

    a = 3.0d0/10.0d0
    b = 3.0d0/5.0d0
    c = 1.0d0/10.0d0

do m=1,4
do i=0,n

	!positive reconstruction at i+1/2
	b0 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
       + 0.25d0*(3.0d0*q(i,m)-4.0d0*q(i+1,m)+q(i+2,m))**2
    b1 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
       + 0.25d0*(q(i-1,m)-q(i+1,m))**2
    b2 = h*(q(i-2,m)-2.0d0*q(i-1,m)+q(i,m))**2 &
       + 0.25d0*(q(i-2,m)-4.0d0*q(i-1,m)+3.0d0*q(i,m))**2

    a0 = a/(eps+b0)**2
    a1 = b/(eps+b1)**2
    a2 = c/(eps+b2)**2

    w0 = a0/(a0+a1+a2)
    w1 = a1/(a0+a1+a2)
    w2 = a2/(a0+a1+a2)

	qL(i,m)=g*w0*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m)) &
           +g*w1*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m)) &
           +g*w2*(2.0d0*q(i-2,m)-7.0d0*q(i-1,m)+11.0d0*q(i,m))


    !negative reconstruction at i+1/2
	b0 = h*(q(i+1,m)-2.0d0*q(i+2,m)+q(i+3,m))**2 &
       + 0.25d0*(3.0d0*q(i+1,m)-4.0d0*q(i+2,m)+q(i+3,m))**2
    b1 = h*(q(i,m)-2.0d0*q(i+1,m)+q(i+2,m))**2 &
       + 0.25d0*(q(i,m)-q(i+2,m))**2
    b2 = h*(q(i-1,m)-2.0d0*q(i,m)+q(i+1,m))**2 &
       + 0.25d0*(q(i-1,m)-4.0d0*q(i,m)+3.0d0*q(i+1,m))**2

    a0 = c/(eps+b0)**2
    a1 = b/(eps+b1)**2
    a2 = a/(eps+b2)**2

    w0 = a0/(a0+a1+a2)
    w1 = a1/(a0+a1+a2)
    w2 = a2/(a0+a1+a2)

    qR(i,m)=g*w0*(11.0d0*q(i+1,m)-7.0d0*q(i+2,m)+ 2.0d0*q(i+3,m))&
           +g*w1*(2.0d0*q(i,m)+5.0d0*q(i+1,m)-q(i+2,m)) &
           +g*w2*(-q(i-1,m)+5.0d0*q(i,m)+2.0d0*q(i+1,m))

end do
end do

return
end



!-----------------------------------------------------------------------------------!
!Time Step
!-----------------------------------------------------------------------------------!
subroutine timestep(nx,ny,dx,dy,cfl,dt,q)
implicit none
integer::nx,ny,i,j
real*8 ::dt,cfl,gamma,dx,dy,smx,smy,radx,rady,p,a,l1,l2,l3
real*8 ::q(-2:nx+3,-2:ny+3,4)

common /fluids/ gamma

!Spectral radius of Jacobian
smx = 0.0d0
smy = 0.0d0

do j=1,ny
do i=1,nx

p = (gamma-1.0d0)*(q(i,j,4)-0.5d0*(q(i,j,2)*q(i,j,2)/q(i,j,1)+q(i,j,3)*q(i,j,3)/q(i,j,1)))
a = dsqrt(gamma*p/q(i,j,1))

!in-x direction
l1=dabs(q(i,j,2)/q(i,j,1))
l2=dabs(q(i,j,2)/q(i,j,1) + a)
l3=dabs(q(i,j,2)/q(i,j,1) - a)
radx = max(l1,l2,l3)

!in-y direction
l1=dabs(q(i,j,3)/q(i,j,1))
l2=dabs(q(i,j,3)/q(i,j,1) + a)
l3=dabs(q(i,j,3)/q(i,j,1) - a)
rady = max(l1,l2,l3)

if (radx.gt.smx) smx = radx
if (rady.gt.smy) smy = rady

end do
end do

dt = min(cfl*dx/smx,cfl*dy/smy)

return
end



!-----------------------------------------------------------------------------------!
!Computing x-fluxes from conserved quantities
!-----------------------------------------------------------------------------------!
subroutine xflux(nx,u,f)
implicit none
integer::nx,i
real*8::gamma,p
real*8::u(0:nx,4),f(0:nx,4)

common /fluids/ gamma

do i=0,nx

p = (gamma-1.0d0)*(u(i,4)-0.5d0*(u(i,2)*u(i,2)/u(i,1)+u(i,3)*u(i,3)/u(i,1)))

f(i,1) = u(i,2)
f(i,2) = u(i,2)*u(i,2)/u(i,1) + p
f(i,3) = u(i,2)*u(i,3)/u(i,1)
f(i,4) = (u(i,4)+ p)*u(i,2)/u(i,1)

end do

return
end

!-----------------------------------------------------------------------------------!
!Computing y-fluxes from conserved quantities
!-----------------------------------------------------------------------------------!
subroutine yflux(ny,u,g)
implicit none
integer::ny,j
real*8::gamma,p
real*8::u(0:ny,4),g(0:ny,4)

common /fluids/ gamma

do j=0,ny

p = (gamma-1.0d0)*(u(j,4)-0.5d0*(u(j,2)*u(j,2)/u(j,1)+u(j,3)*u(j,3)/u(j,1)))

g(j,1) = u(j,3)
g(j,2) = u(j,3)*u(j,2)/u(j,1)
g(j,3) = u(j,3)*u(j,3)/u(j,1) + p
g(j,4) = (u(j,4)+ p)*u(j,3)/u(j,1)
end do

return
end
