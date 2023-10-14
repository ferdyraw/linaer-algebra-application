# Linear Algebra Application in Python
The aim of this project is to make a Linear Equation System (SPL) completion program,
prove the diagonalized matrix, and look for Singular Value Decomposition (SVD) in python 
using the Gauss elimination method and/or Gauss-Jordan. SPL can have a single solution, 
multiple solutions, or no solution. Looking for Eigen values and Eigen Vectors to prove
matrix diagonalization and solving complex linear equation systems with SVD.

SPL can be solved numerically by the Gaussian elimination method and the elimination
method Gauss-Jordan. The program must be able to handle the following cases:
a) If SPL has a unique solution, show the solution
b) If SPL has infinite solutions, display the solutions in the form of parameters
c) If SPL has no solution, write there is no solution

## Case Study (Menu)
1. Solve Linear Equation System 
   a) Normal Method
   b) Least-squares Method
   c) Gaussian Elimination Method
2. Solve Linear Equation System 
   a) Normal Method
   b) Least-squares Method
   c) Gaussian Elimination Method
3. Find :
   - Polynomical Characteristic
   - Eigenvalues and eigenvectors
   - Diagonalized
4. Singular Value Decomposition (SVD)
5. SVD Complex

## Input Format
### Case Study 1-4 (example) :
Rows = 4
Columns = 4

A = 
1 1 -1 -1
2 5 -7 -5
2 -1 1 3
5 2 -4 2

b = 
1 -2 4 6

### Case Study 5 :
Rows = 6
Columns = 5

A = 
0+1j 0+0j -1+0j 0+0j 1+0j
1+0j 1+0j 0-1j 0+0j 0+0j
2+0j 0+0j 0+5j 0+0j 0+0j
0+0j 0+2j 2+0j 0+0j 0+0j
0+0j 0+0j 0+0j 1+0j 1+0j
0+1j 0+1j 0+0j 0+0j 0+0j

b = 
3+0j 0+4j 0-3j -5+0j 5+0j -3+0j

## Output 
The results will be shown in the terminal and saved in some txt files.