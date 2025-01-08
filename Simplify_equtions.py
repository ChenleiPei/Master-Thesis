import sympy as sp
import os


def simplify_expression(expression):
    try:
        # transform the expression into a sympy expression
        expr = sp.sympify(expression)

        # use the simplify function to simplify the expression
        simplified_expr = sp.simplify(expr)

        return simplified_expr
    except Exception as e:
        # if an error occurs, return the error message
        return f"Error simplifying expression: {expression}. {str(e)}"


def simplify_expressions_in_file(filename):
    # get the base filename
    base_filename = os.path.basename(filename)

    # create a new filename
    new_filename = f"simplified_{base_filename}"

    # open the new file
    with open(new_filename, 'w') as output_file:
        # open the file containing the equations
        with open(filename, 'r') as file:
            for line in file:
                # delete leading and trailing whitespaces
                expression = line.strip()

                simplified = simplify_expression(expression)

                # save the results
                output_line = f"{simplified}\n"
                output_file.write(output_line)

    return new_filename


# file containing equations
filename = "equations_20.txt"

# do the simplification
new_file_path = simplify_expressions_in_file(filename)
print(f"Results saved in: {new_file_path}")
