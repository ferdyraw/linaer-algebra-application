import numpy as np

class Matrix:
    def __init__(self, n, m, A, b):
        self.n = n
        self.m = m
        self.A = A
        self.b = b

    def svd(self):
        # Solving the matrix using SVD
        U, x, V = np.linalg.svd(self.A)
        U = limit(U)
        x = limit(x)
        V = limit(V)
        
        print("\nU :")
        print(U)
        matrix_to_txt('U', U, 'no4.txt')
        
        print("\nSVD :")
        print(x)
        matrix_to_txt('SVD', x, 'no4.txt')
        
        print("\nV :")
        print(V)
        matrix_to_txt('V', V, 'no4.txt')
        
        print("\nThe results have been saved in no4.txt !")
    
    def svd_complex(self):
        # Solving the equatoins using SVD Complex
        U, s, V = np.linalg.svd(self.A, full_matrices=False)
        x = np.dot(V.T, np.dot(np.diag(1 / s), np.dot(U.T, self.b)))

        print("\nSVD Complex :")
        for i, sol in enumerate(x):
            print(f"x{i+1} = {sol}")
    
        matrix_to_txt('SVD Complex', x, 'no5.txt')        
        print("\nThe results have been saved in no5.txt !")
    
    def poly_eigen_diag(self):
        # Find polynomial characteristic, eigenval, eigenvecs, and diagonalize
        characteristic_poly = find_characteristic_poly(self.A)
        eigenvalues, eigenvectors = find_eigen(self.A)
        P, D, diagonalize = find_diagonalize(self.A)
        
        eigenvalues = limit(eigenvalues)
        eigenvectors = limit(eigenvectors)

        print("\nPolynomial Characteristic:")
        print(characteristic_poly)
        matrix_to_txt('Polynomial Characteristic', characteristic_poly, 'no3.txt')

        print("\nEigenvalues:")
        print(eigenvalues)
        matrix_to_txt('Eigenvalues', eigenvalues, 'no3.txt')

        print("\nEigenvectors:")
        print(eigenvectors)
        matrix_to_txt('Eigenvectors', eigenvectors, 'no3.txt')

        print("\nP:")
        print(P)
        matrix_to_txt('P', P, 'no3.txt')

        print("\nD:")
        print(D)
        matrix_to_txt('D', D, 'no3.txt')

        print("\nDiagonalized A:")
        print(diagonalize)
        matrix_to_txt("Diagonalized A", diagonalize, 'no3.txt')
        
        print("\nThe results have been saved in no3.txt !")
        
    def spl(self, sk):
        
        print("\nMenu :")
        print("a. Normal Method")
        print("b. Least-squares Method")
        print("c. Gaussian Elimination Method")
        
        menu = input("Pilih : ")
        if menu == 'a':
            print("\nNormal Method :")
            try:
                solutions = limit(np.linalg.solve(self.A, self.b))
                for i, sol in enumerate(solutions):
                    print(f"x{i+1} = {sol}")
                matrix_to_txt('Normal Method', solutions, 'no{}.txt'.format(sk))
            except np.linalg.LinAlgError:
                print("No solution!")
                with open('no{}.txt'.format(sk), 'a') as file:
                    file.write("\nNormal Method :\n")
                    file.write("No solution!\n")
        elif menu == 'b':
            print("\nLeast-squares Method :")
            try:
                solutions, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
                solutions = limit(solutions)
                for i, sol in enumerate(solutions):
                    print(f"x{i+1} = {sol}")
                matrix_to_txt('Least-squares Method', solutions, 'no{}.txt'.format(sk))
            except np.linalg.LinAlgError:
                print("No solution!")
                with open('no{}.txt'.format(sk), 'a') as file:
                    file.write("\nLeast-squares Method :\n")
                    file.write("No solution!\n")
        else:
            solutions = gaussian_elimination(self.A, self.b)
        
            print("\nGaussian Elimination :")
            if type(solutions) == str:
                print(solutions)
                with open('no{}.txt'.format(sk), 'a') as file:
                    file.write("\nGaussian Elimination :\n")
                    file.write(f"{solutions}\n")
            else:
                solutions = limit(solutions)
                for i, sol in enumerate(solutions):
                    print(f"x{i+1} = {sol}")
                matrix_to_txt('Gaussian Elimination Method', solutions, 'no{}.txt'.format(sk))
        
        print("\nThe results have been saved in no{}.txt !".format(sk))

def find_characteristic_poly(matrix):
    # Find polynomial characteristics
    char_poly = np.poly(matrix)
    return limit(char_poly)

def find_eigen(matrix):
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def find_diagonalize(matrix):
    # Find diagonalize matrix using eigen decomposition
    eigenvalues, eigenvectors = find_eigen(matrix)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)

    # Compute P^(-1)AP
    diagonalized_A = P_inv @ matrix @ P

    return limit(P), limit(D), limit(diagonalized_A)

def gaussian_elimination(matrix, vector):
    n = len(matrix)
    m = len(matrix[0])

    for i in range(n):
        if matrix[i][i] == 0:
            for j in range(i+1, n):
                if matrix[j][i] != 0:
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    vector[i], vector[j] = vector[j], vector[i]
                    break

        if matrix[i][i] == 0:
            if vector[i] == 0:
                return "Infinite solutions!"
            else:
                return "No solution!"

        pivot = matrix[i][i]
        for j in range(i, m):
            matrix[i][j] /= pivot
        vector[i] /= pivot

        for j in range(n):
            if j != i:
                factor = matrix[j][i]
                for k in range(i, m):
                    matrix[j][k] -= factor * matrix[i][k]
                vector[j] -= factor * vector[i]

    return vector

def limit(matrix):
    return np.round(matrix, 5)

def matrix_to_txt(title, matrix, file_name):
    with open(file_name, 'a') as file:
        file.write(f"\n{title} :\n")
        if file_name[2] == '1' or file_name[2] == '2' or file_name[2] == '5':
            for i, sol in enumerate(matrix):
                file.write(f"x{i+1} = {sol}\n")
        else:
            for line in matrix:
                file.write(f"{line}\n")

print("Linear Algebra Application in Python")

sk = int(input("Case study : "))
n = int(input("\nEnter the number of rows : "))
m = int(input("Enter the number of columns : "))

A = None
b = None

if sk == 5:
    A = np.zeros((n, m), dtype=np.complex_)
    b = np.zeros(n, dtype=np.complex_)
else:
    A = np.zeros((n, m))
    b = np.zeros(n)

print("\nEnter the matrix A (or coefficients) :")
for i in range(n):
    if sk != 5:
        A[i] = list(map(float, input().split()))
    else:
        row_i = input()
        A[i] = [complex(x.strip()) for x in row_i.split()] 

matrix_to_txt('A', A, 'no{}.txt'.format(sk))

if sk < 3:
    b = list(map(float, input("\nEnter the matrix b (constants) :\n").split()))
    matrix_to_txt('b', b, 'no{}.txt'.format(sk))
elif sk == 5:
    y = input("\nEnter the matrix b (constants) :\n")
    b = [complex(x.strip()) for x in y.split()]
    matrix_to_txt('b', b, 'no{}.txt'.format(sk))

matrix = Matrix(n, m, A, b)
if sk == 1 or sk == 2:
    matrix.spl(sk)
elif sk == 3:
    matrix.poly_eigen_diag()
elif sk == 4:
    matrix.svd()
else:
    matrix.svd_complex()