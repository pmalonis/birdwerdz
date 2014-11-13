#include <math.h>
#include <stdarg.h>
typedef double DTYPE_t;

void
distance (DTYPE_t* a, DTYPE_t* b, DTYPE_t* dist_mat_ptr, int length) {
        
        DTYPE_t dot_prod = 0;
        DTYPE_t norm_a = 0;
        DTYPE_t norm_b = 0;
        int i;
        for (i = 0; i < length; i++) {
                dot_prod += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];                       
        }
        
        *dist_mat_ptr =  1-(dot_prod/sqrt(norm_a * norm_b));
 
}



/* void */
/* assign_D (DTYPE_t* distance, DTYPE_t* D_entry, */
/*           DTYPE_t* T_entry, int nargs, ... ) { */
        
/*         va_list arguments; */
/*         DTYPE_t min = 0; */
/*         DTYPE_t argmin = 0; */
/*         va_start(arguments, nargs); */
/*         int i; */
/*         DTYPE_t arg; */
/*         for (i = 0; i < nargs; i++) { */
/*                 arg=va_arg(arguments, DTYPE_t); */
/*                 if ((0-arg) > (0-min)) { */
/*                         min = arg; */
/*                         argmin = i; //algorithm doesn't use 0-based indexing for argmin */
/*                 } */
/*         } */
/*         va_end(arguments); */
/*         *D_entry = min + *distance; */
/*         *T_entry = argmin + 1; */
         
/* } */
 
void
assign_D (DTYPE_t* distance, DTYPE_t* D_entry,
          int* T_entry, DTYPE_t arg_1, DTYPE_t arg_2, DTYPE_t arg_3) {
        
        if (arg_1 <= arg_2 && arg_1 <= arg_3) {
                *D_entry = *distance + arg_1;
                *T_entry = 1;
        }
        else if (arg_2 <= arg_1 && arg_2 <= arg_3) {
                *D_entry = *distance + arg_2;
                *T_entry = 2;              
        }
        else {
                *D_entry = *distance + arg_3;
                *T_entry = 3;
        }
                
}
