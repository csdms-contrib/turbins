static char help[] = "3-Dimensional Gravity Current Simulation, with Variable Geometry.\n\n";
#include "definitions.h"
#include "DataTypes.h"
#include "Velocity.h"
#include "Conc.h"
#include "Pressure.h"
#include "Grid.h"
#include "Output.h"
#include "Display.h"
#include "Debugger.h"
#include "Outflow.h"
#include "Inflow.h"
#include "Surface.h"
#include "Immersed.h"
#include "Memory.h"
#include "Input.h"
#include "Resume.h"
#include "Communication.h"
#include "MyMath.h"
#include "Output.h"
#include "ENO.h"
#include "Extract.h"
#include "gvg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GET_TIMES
#undef SERIOUS_DEBUGGING
#undef SERIOUS_DEBUGGING_NOW
#undef DEBUGGING_FIND_MAX

double W_total_Poisson_solver_time=0.0;
double W_total_writing_time=0.0;
double W_total_external_comm_time=0.0;
double W_total_uvwc_solution_time=0.0;
double W_total_rhs_time= 0.0;
double W_total_convective_time=0.0;
double W_total_ENO_time=0.0;
double W_total_convective_setup_time=0.0;
double W_total_ifsolid1_time=0.0;
double W_total_ifsolid2_time=0.0;
double W_total_ifsolid3_time=0.0;
double W_total_ENO_deposition_time=0.0;
double W_total_ENO_setup_time=0.0;
PetscLogDouble T_Start, T_End;

// If memory profiling is required these values will be updated. (bytes)
//See definitions.h
#ifdef MEMORY_PROFILING
double mem1=0.0;
double mem2=0.0;
int grid_mem=0;
int u_mem=0;
int v_mem=0;
int w_mem=0;
int c_mem=0;
int p_mem=0;
int u_solver_mem=0;
int v_solver_mem=0;
int w_solver_mem=0;
int c_solver_mem=0;
int p_solver_mem=0;
#endif

int main(int argc, char **args) {

    int iter = 0;
    int NConc, iconc;
    int j;
    int essential_resume_data;
    Velocity *u, *v, *w;
    Pressure *p;
    Concentration **c;
    Parameters *params;
    MAC_grid *grid;
    GVG_bag *data_bag;
    Resume *resume;
    PetscMPIInt numprocs, rank;
    PetscErrorCode ierr;

    PetscInitialize(&argc, &args,(char *)0, help);
#ifdef MEMORY_PROFILING
    //PetscMemorySetGetMaximumUsage();
#endif

    /* Get the number of processors and rank of that */
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &numprocs); PETScErrAct(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); PETScErrAct(ierr);

    PetscPrintf(PCW, "\n*************************************\n");
    PetscPrintf(PCW, "  gvg3D is running on %d processor(s)\n", numprocs);
    PetscPrintf(PCW, "*************************************\n\n");

    params = (Parameters *)malloc(sizeof(Parameters));
    Memory_check_allocation(params);

    /* Number of processors */
    params->size = numprocs;
    /* Current Processor rank */
    params->rank = rank;

    /* Read input parameters from "input.inp" and set the other parameters */
    for (j=0; j<numprocs; j++) {
        if (j == rank) {

            Input_read_parameters_from_file(params);
        }
        MPI_Barrier(PCW);
    }
    Input_set_internal_parameters(params);

    for (j=0; j<numprocs; j++) {
        if (j == rank) {

            //Display_parameters(params);
        }
        MPI_Barrier(PCW);
    }

    NConc = params->NConc;
    /*******************************************************************/

    /* Allocate memory for all the simulation data.
    Also, create a data_bag which holds the pointer to the simulation data */
    data_bag = GVG_create_bag(params);
    u = data_bag->u;
    v = data_bag->v;
    w = data_bag->w;
    p = data_bag->p;
    c = data_bag->c;
    grid = data_bag->grid;

    /* create resume read/writer */
    /* In case of reader, it reads the data from the previous runs and resumes the simulation */
    /* In case of write, it writes essential data needed for future restarts */

    essential_resume_data  = 3; /* data: u, v, w */
    essential_resume_data += 3; /* dpdx, dpdy, dpdz */
    /* Only store/read grid_c data, if the geometry could change due to erosion or deposition */
    if (!params->constant_geometry) {

        essential_resume_data += 1; /* c_is_solid */
    }
    essential_resume_data += NConc; /* concentration fields */

    resume = Resume_create(essential_resume_data);


    /* if the simulation is starting from t=0 */
    if (!params->resume) {

        /*use MAC_grid to identify which cells are fluid, and which are solid*/
        GVG_identify_geometry(grid, params);

        /* This function initializes the inflow and initial configuration of the concentration fields */
        GVG_initialize_primitive_data(u, v, w, c, grid, params);
    } else {


        if (params->constant_geometry) { /* Bottom geometry is not changing due to erosion or deposition */

            /* Tell resumer not to read grid_c data */
            Resume_reset_IO_grid_flag(resume);
            GVG_read_previous_simulation_data(resume, data_bag);
            /*use MAC_grid to identify which cells are fluid, and which are solid*/
            GVG_identify_geometry(grid, params);

        } else { /* Inform resumer to read grid_c from file. Do not use this. Incomplete */

            /* Inform resumer to read grid_c data */
            Resume_set_IO_grid_flag(resume);
            GVG_read_previous_simulation_data(resume, data_bag);
            /* Redefine u,v,w grid based on c-grid. Here */

        } /* else */
    } /* else */

    /* Setup immersed boundary nodes and coefficients to impose no-slip B.C. for the velocity field on the solid boundary */
    Immersed_setup_q_immersed_nodes(grid, 'u');
    PetscPrintf(PCW, "gvg.c/ u-immersed has been setup successfully\n");
    Immersed_setup_q_immersed_nodes(grid, 'v');
    PetscPrintf(PCW, "gvg.c/ v-immersed has been setup successfully\n");
    Immersed_setup_q_immersed_nodes(grid, 'w');
    PetscPrintf(PCW, "gvg.c/ w-immersed has been setup successfully\n");
    Immersed_setup_q_immersed_nodes(grid, 'c');
    PetscPrintf(PCW, "gvg.c/ c-immersed has been setup successfully\n");
    PetscPrintf(PCW, "gvg.c/ Immersed boundary has been setup successfully\n");

    /* Grid status */
    /*
 Display_DA_3D_data(grid->L_u_status, grid, params, "ugrid", 'u');
 Display_DA_3D_data(grid->L_u_status, grid, params, "vgrid", 'v');
 Display_DA_3D_data(grid->L_u_status, grid, params, "wgrid", 'w');
 Display_DA_3D_data(grid->L_u_status, grid, params, "cgrid", 'c');
*/	

    /* Setup the linear system for primitive variables based on the fluid nodes */
    GVG_setup_lsys_accounting_geometry(u, v, w, p, c, grid, params);

    /* Shift the diagonal part of the matrices with 1/dt_old */
    if (params->resume) {

        Velocity_modify_diagonal(u, grid, params, data_bag->dt_old, 0.0);
        Velocity_modify_diagonal(v, grid, params, data_bag->dt_old, 0.0);
        Velocity_modify_diagonal(w, grid, params, data_bag->dt_old, 0.0);

        for (iconc=0; iconc<NConc; iconc++) {

            Conc_modify_diagonal(c[iconc], grid, data_bag->dt_old, 0.0);
        }
    } /* if */

    /* Display grid information on each processor */
    Display_DA_3D_info(grid, params) ;

    /* for debugging purposes */
    /*
 //Debugger_set_velocity_data(u, grid) ;
 Debugger_set_velocity_RHS(u, grid) ;
 Debugger_set_pressure_RHS(p, grid) ;
    Debugger_solve_p_Poisson_equation(p, grid) ;
    Debugger_solve_vel_Poisson_equation(u, grid);
 */

    //Debugger_set_q_rhs(u, c[0], p, grid, params, 'p') ;
    //Pressure_solve(p);
    //Debugger_validate_q(u, c[0], p, grid, params, 'p') ;
    //getchar();
    /* Debugging */
    /*
 printf("gvg.c/ u-solution validation\n");
 Debugger_set_q_rhs(u, c[0], p, grid, params, 'u') ;
 Velocity_solve(u);
 Debugger_validate_q(u, c[0], p, grid, params, 'u') ;
 printf("gvg.c/ v-solution validation\n");
 Debugger_set_q_rhs(v, c[0], p, grid, params, 'v') ;
 Velocity_solve(v);
 Debugger_validate_q(v, c[0], p, grid, params, 'v') ;
 printf("gvg.c/ u-solution validation\n");
 Debugger_set_q_rhs(w, c[0], p, grid, params, 'w') ;
 Velocity_solve(w);
 Debugger_validate_q(w, c[0], p, grid, params, 'w') ;
 Debugger_set_q_rhs(u, c[0], p, grid, params, 'c') ;
 Conc_solve(c[0]);
 Debugger_validate_q(u, c[0], p, grid, params, 'c') ;
*/
/*
    Debugger_set_velocity_RHS(u, grid) ;
    Velocity_solve(u);
    Debugger_validate_q(u, NULL, NULL,grid, params, 'u');
    Debugger_set_velocity_RHS(v, grid) ;
    Velocity_solve(v);
    Debugger_validate_q(v, NULL, NULL,grid, params, 'v');

    Velocity_cell_center(u, v, w, grid, params);
    Communication_update_ghost_nodes(&grid->DA_3D, &u->G_data, &u->L_data, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &v->G_data, &v->L_data, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &w->G_data, &w->L_data, 'I');

    Extract_viscous_dissipation_rate(u, v, w, grid, params);
    PetscPrintf(PCW, "gvg.c/ Stokes diss rate: %f\n", u->W_dissipation_rate);
    getchar();
*/

    /*perform 3rd order  Runge-Kutta timestepping to advance to t_final*/
    iter = GVG_tvd_rk(data_bag, resume);

    PetscPrintf(PCW, "******************************************\n");
    PetscPrintf(PCW, "Simulation complete, took %d iterations.\n", iter);
    PetscPrintf(PCW, "******************************************\n");


    /* Now, release the allocated memory for the variables */
    Velocity_destroy(u);
    Velocity_destroy(v);
    Velocity_destroy(w);
    Pressure_destroy(p);

    for(iconc=0; iconc <NConc; iconc++) {

        Conc_destroy(c[iconc], params, iconc);
    } /* for iconc */
    free(c);

    Grid_destroy(grid);
    Input_destroy_parameters(params);
    free(data_bag);
    Resume_destroy(resume);

    ierr = PetscFinalize(); PETScErrAct(ierr);

    return 0;
}
/***************************************************************************************************/

/* This function allocates memory for all the simulation data and then wrap them in a bag */
/* This function creates a general bag which holds the pointer to all important variables and some useful data */
GVG_bag *GVG_create_bag(Parameters *params) {

    GVG_bag *new_bag;
    MAC_grid *grid;
    new_bag = (GVG_bag *)malloc(sizeof(GVG_bag));
    Memory_check_allocation(new_bag);

    new_bag->params = params;
    /*******************************************************************/
    /* First, create primitive variables based on the input parameters */
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    new_bag->grid = Grid_create(params);
    PetscPrintf(PCW, "Grid has been created successfully...\n");
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    grid_mem += (int)(mem2 - mem1);
#endif
    grid = new_bag->grid;

#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    new_bag->u = Velocity_create(grid, params, 'u');
    PetscPrintf(PCW, "u-Velocity has been created successfully...\n");
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    u_mem += (int)(mem2 - mem1);
#endif

#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    new_bag->v = Velocity_create(grid, params, 'v');
    PetscPrintf(PCW, "v-Velocity has been created successfully...\n");
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    v_mem += (int)(mem2 - mem1);
#endif

#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    new_bag->w = Velocity_create(grid, params, 'w');
    PetscPrintf(PCW, "w-Velocity has been created successfully...\n");
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    w_mem += (int)(mem2 - mem1);
#endif

#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    new_bag->p = Pressure_create(grid, params) ;
    PetscPrintf(PCW, "Pressure has been created successfully...\n");
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    p_mem += (int)(mem2 - mem1);
#endif

    /* Create concentration field */
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem1);
#endif
    int NConc = params->NConc;
    new_bag->c = (Concentration **)malloc(NConc * sizeof(Concentration *));
    Memory_check_allocation(new_bag->c);

    int iconc;
    for (iconc=0; iconc<NConc; iconc++) {

        new_bag->c[iconc] = Conc_create(grid, params, iconc);
        PetscPrintf(PCW, "Concentration field[%d] has been created successfully...\n", iconc);
    }
#ifdef MEMORY_PROFILING
    PetscMallocGetCurrentUsage(&mem2);
    c_mem += (int)(mem2 - mem1);
#endif
    /*******************************************************************/

    return (new_bag);
}
/***************************************************************************************************/

/* This function reads the previous simulation data and resumes from the last stopped time. */
void GVG_read_previous_simulation_data(Resume *resume, GVG_bag *data_bag) {

    int NConc, iconc;
    int set_data_flag;

    NConc = data_bag->params->NConc;

    /* Tell the resumer just to read data from file */
    Resume_set_read_flag(resume);
    /* Set writer flag off */
    Resume_reset_write_flag(resume);

    /* Set the pointers to primitive variables */
    Resume_set_data(resume, data_bag);

    /* read primary data, e.g. dt, dt_old, iteration, time, ... from ASCII file */
    Resume_read_primary_data(resume, data_bag);

    /* Now, read all the data in parallel vectors */
    Resume_read_data(resume);

    if (data_bag->params->outflow) {

        /* Copy outflow data from G_data 3d data array (stored in the last yz plane) into outflow[][] (2D
  array*/
        Outflow_vel_copy_data_into_2D_array(data_bag->u, data_bag->grid);
        Outflow_vel_copy_data_into_2D_array(data_bag->v, data_bag->grid);
        Outflow_vel_copy_data_into_2D_array(data_bag->w, data_bag->grid);

        for (iconc=0; iconc<NConc; iconc++) {

            Outflow_concentration_copy_data_into_2D_array(data_bag->c[iconc], data_bag->grid) ;
        }

        /* Now, set the last plane yz data in G_data 3d array equal to zero */
        set_data_flag = NO;
        Outflow_vel_copy_data_into_last_plane(data_bag->u, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->v, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->w, data_bag->grid, set_data_flag);

        for(iconc=0; iconc<NConc; iconc++) {

            Outflow_concentration_copy_data_into_last_plane(data_bag->c[iconc], data_bag->grid, set_data_flag);
        } /* for iconc */

    } /* if data_bag */


    if (data_bag->params->sedimentation) {

        char filename[FILENAME_MAX];
        int NZ = data_bag->grid->NZ;
        int NX = data_bag->grid->NX;

        for(iconc=0; iconc<NConc; iconc++) {

            if (data_bag->c[iconc]->Type == PARTICLE) {

                sprintf(filename, "Resume_sed_height_c%d_data.bin", iconc);
                double **data = NULL;

                data = data_bag->c[iconc]->G_deposit_height;

                Resume_read_2D_data(resume, data, filename, NX, NZ);
                Conc_reset_nonlocal_deposit_height(data_bag->c[iconc], data_bag->grid);
            } /* if */
        } /* for iconc */
    }


}
/***************************************************************************************************/

/* This function sets up the linear system LHS matrices for all primitive variables using the defined grid (geometry) */
void GVG_setup_lsys_accounting_geometry(Velocity *u, Velocity *v, Velocity *w, Pressure *p, Concentration **c, MAC_grid
                                        *grid, Parameters *params) {

    int iconc;
    int NConc;

    NConc = params->NConc;

    /*set up linear system solvers, based on which cells contain fluid, and which are solid*/
    Velocity_setup_lsys_accounting_geometry(u, grid, params);
    PetscPrintf(PCW, "Velocity u linear system has been set-up successfully...\n");

    Velocity_setup_lsys_accounting_geometry(v, grid, params);
    PetscPrintf(PCW, "Velocity v linear system has been set-up successfully...\n");

    Velocity_setup_lsys_accounting_geometry(w, grid, params);
    PetscPrintf(PCW, "Velocity w linear system has been set-up successfully...\n");

    Pressure_setup_lsys_accounting_geometry(p, grid, params);
    PetscPrintf(PCW, "Pressure linear system has been set-up successfully...\n");

    for (iconc=0; iconc<NConc; iconc++) {

        Conc_setup_lsys_accounting_geometry(c[iconc], grid, params);
        PetscPrintf(PCW, "Concentration field %d's linear system has been set-up successfully...\n", iconc);
    }

#ifdef MEMORY_PROFILING
    Display_matrix_info(u->A, "matrix u") ;
    Display_matrix_info(v->A, "matrix v") ;
    Display_matrix_info(w->A, "matrix w") ;
    Display_matrix_info(c[0]->A, "matrix c") ;
    Display_matrix_info(p->A, "matrix p") ;
    getchar();
#endif


}
/***************************************************************************************************/

/* This function initializes the inflow and initial configuration of the concentration fields */
void GVG_initialize_primitive_data(Velocity *u, Velocity *v, Velocity *w, Concentration **c, MAC_grid *grid,
                                   Parameters *params) {

    int iconc;
    int NConc;

    NConc = params->NConc;

    if (params->inflow) {

        Inflow_u_velocity_profile(u, grid, params, 0.0);
        PetscPrintf(PCW, "u-Velocity inflow profile has been set successfully...\n");
    }

    for (iconc=0; iconc<NConc; iconc++) {

        if (params->inflow) {

            Inflow_conc_profile( c[iconc], grid, params, 0.0);
            PetscPrintf(PCW, "Concentration field %d's inflow profile has been set successfully...\n");
        }
        else {

            Conc_initialize( c[iconc], grid, params );
            PetscPrintf(PCW, "Concentration field %d has been initialized successfully...\n", iconc);
        }

        //Conc_update_boundary( c[iconc], grid, params ); /*enforce suitable boundary*/
        PetscPrintf(PCW, "Concentration field %d's boundaries have been updated successfully...\n", iconc);
    }

    short int initial_nonzero_velocity = NO;
    if (initial_nonzero_velocity) {

        Velocity_nonzero_initialize(u, grid, params);
        Velocity_nonzero_initialize(v, grid, params);
        Velocity_nonzero_initialize(w, grid, params);
    } /* if */
}
/***************************************************************************************************/

/* This function integrates the u,v and conc transport equation in time up to time_max using 2nd or 3rd order TVD-Runge Kutta method */
int GVG_tvd_rk(GVG_bag *data_bag, Resume *resume) {

    double time , max_time, dt, dt_old;
    double output_time, output_time_step;
    int iter, output_iter;
    double G_div_max;
    int NConc, iconc;
    ENO_Scheme *ENO;
    Output *output;
    int ierr;
    double resume_writer_time;
    Velocity *u, *v, *w;
    Pressure *p;
    Concentration **c;
    MAC_grid *grid;
    Parameters *params;
    PetscLogDouble T1, T2;
#ifdef TVD_RK3 /* third order */
    double tvd_coef1 = 0.75;
    double tvd_coef2 = 0.25;
#endif
#ifdef TVD_RK2 /* second order */
    double tvd_coef1 = 0.5;
    double tvd_coef2 = 0.5;
#endif

    u = data_bag->u;
    v = data_bag->v;
    w = data_bag->w;
    p = data_bag->p;
    c = data_bag->c;

    grid   = data_bag->grid;
    params = data_bag->params;


    /*******************/
    /* Get the time properties */
    max_time         = params->time_max;
    output_time_step = params->output_time_step;

    /* set primary data from previous simulation */
    if (params->resume) {

        dt     = data_bag->dt;
        dt_old = data_bag->dt_old;
        time   = data_bag->time;
        iter   = data_bag->iter;
        output_iter = data_bag->output_iter;
        output_time = output_iter*output_time_step;
        resume_writer_time = time + params->resume_saving_timestep;

	PetscPrintf(PCW, "gvg.c/ Resume info read: \n"); 
	PetscPrintf(PCW, "gvg.c/    dt:%f dt_old:%f time:%f iter:%d output_iter:%d \n", dt, dt_old, time, iter, output_iter); 
    } else { /* Start from t=0 and default values */

        dt_old = 0.0;
        dt     = params->default_dt;

        output_time = 0.0;
        time        = 0.0;
        output_iter = 0;
        iter        = 1;
        /* Time step which data should be saved for future runs */
        resume_writer_time = params->resume_saving_timestep;

    } /* else */

    /* update the data_bag information */
    data_bag->dt          = dt;
    data_bag->dt_old      = dt_old;
    data_bag->iter        = iter;
    data_bag->time        = time;
    data_bag->output_iter = output_iter;

    NConc = params->NConc;

    /* This data structure, holds all the variables for the ENO scheme used for calculating the convective terms */
    ENO = ENO_create(params);
    PetscPrintf(PCW, "ENO Scheme variables have been created successfully...\n");

    /* writer object which is used to write runtime data */
    output = Output_create(grid, params);
    PetscPrintf(PCW, "Output writer has been created successfully...\n");

    Output_immersed_info(output, grid, params, 'u');
    Output_immersed_info(output, grid, params, 'v');
    Output_immersed_info(output, grid, params, 'w');
    Output_immersed_info(output, grid, params, 'c');
    Output_immersed_control(output, grid, params, 'u');
    Output_immersed_control(output, grid, params, 'v');
    Output_immersed_control(output, grid, params, 'w');
    Output_immersed_control(output, grid, params, 'c');
    //Output_immersed_control_new(output, grid, params, 'c');
    PetscPrintf(PCW, "gvg.c/ Immersed boundary info has been written to file(s) successfully\n");

    //Output_binary_3D_surface(output, grid->surf, params, 0);
    Output_surface(output, grid->surf, params, 0);
    Surface_destroy(grid->surf, grid);
    grid->surf = NULL;
    PetscPrintf(PCW, "gvg.c/ Solid surface has been written to file successfully. It is also destroyed...\n");

    /* Based on the number of processors, write the grid partitioning among processors to file "GridPartition.txt" */
    Output_write_grid_partition_info(output, grid, params) ;
    PetscPrintf(PCW, "Grid partition has been written to file successfully...\n");

    /*initial output to file. Corresponding to zero velocity. */
    Velocity_cell_center(u, v, w, grid, params);

    /* Saving the initial conditions */
    if (!params->resume) {

        Output_flow_properties(output, data_bag);
        output_time += output_time_step;
        output_iter++;
        PetscPrintf(PCW, "Initial flow properties have been saved successfully...\n");
    } /* if */

    PetscPrintf(PCW, "\n*** Beginning main simulation loop\n");

    /* Since LHS matrix is constant, no update is needed for the preconditioner */
    /* The solution is stored in data array */

    ierr = PetscGetTime(&T_Start);PETScErrAct(ierr);

    while (time <= max_time) {

        /* update the data_bag information */
        data_bag->dt          = dt;
        data_bag->dt_old      = dt_old;
        data_bag->iter        = iter;
        data_bag->time        = time;
        data_bag->output_iter = output_iter;

        if (time >= output_time) {

#ifdef GET_TIMES
            ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

            if (params->p_output) {

                //find_real_pressure(p, grid, params);
            }
            //Output_flow_properties(output, u, v, w, p, c, grid, params, output_iter, time);
            Velocity_cell_center(u, v, w, grid, params);
            Output_flow_properties(output, data_bag) ;

            output_time += output_time_step;
            output_iter++;
            PetscPrintf(PCW, " ***************************************************     \n");
            PetscPrintf(PCW, "     Runtime data has been saved successfully\n");
            PetscPrintf(PCW, " ***************************************************     \n");

            G_div_max = Pressure_compute_velocity_divergence(p, u, v, w, grid, params);
            PetscPrintf(PCW, " Maximum Velocity Divergence: %2.2e\n", G_div_max);

#ifdef GET_TIMES
            ierr = PetscGetTime(&T2);PETScErrAct(ierr);
            W_total_writing_time += T2 - T1;
#endif
        }

        if (fabs(time-resume_writer_time) < 1.0e-6) {

#ifdef GET_TIMES
            ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

            /* update the data_bag information */
            data_bag->dt          = dt;
            data_bag->dt_old      = dt_old;
            data_bag->iter        = iter;
            data_bag->time        = time;
            data_bag->output_iter = output_iter;

            GVG_write_workspace_data(data_bag, resume);
            PetscPrintf(PCW, "Essential data has been saved for future simulations, successfully\n");
#ifdef GET_TIMES
            ierr = PetscGetTime(&T2);PETScErrAct(ierr);
            W_total_writing_time += T2 - T1;
#endif

            resume_writer_time += params->resume_saving_timestep;
        } /* if resume */


        /* find dt according to cfl condition and next output write time step */
        if (iter>1) {

            /* First, find the velocities at the cell center */
            Velocity_cell_center(u, v, w, grid, params);
            dt = GVG_new_time_step(u, v, w, grid, params, time, output_time, dt_old);
		PetscPrintf(PCW, "gvg.c/ dt found using CFL condition:%f\n", dt); 		
        } /* if */

        /*keep track of old velocity to perform RK averaging*/
        GVG_store_flow_variables_old_data(u, v, w, c, params);

        /*****************************************/
        /*1st RK step, advance time to t+dt */
        /* Update inflow profiles if we have transient inflow */
        if (params->transient_inflow) {

            Inflow_u_velocity_profile( u, grid, params, time);
            for (iconc=0; iconc<NConc; iconc++) {

                Inflow_conc_profile(c[iconc], grid, params, time);
            }
        }

        /*find convective outflow boundary values, at time t+dt*/
        if (params->outflow) {

            Outflow_impose_convective_boundary( u, v, w, c, grid, params, dt) ;

            //Display_2D_outflow(u->G_outflow, grid, params, "After:u-outflow", 'u');
            //getchar();
        }

        GVG_integrate_all_the_equations_in_time(u, v, w, p, c, grid, ENO, params, dt, dt_old, 1);

        /*****************************************/
        /* 2nd RK step, advance time to t+2dt */

        /*find convective outflow boundary values, at time t + 2*dt*/
        if (params->outflow) {

            Outflow_impose_convective_boundary( u, v, w, c, grid, params, dt) ;
        }

        GVG_integrate_all_the_equations_in_time(u, v, w, p, c, grid, ENO, params, dt, dt_old, 2);

        /* Now have vel_n and vel_n+2, average to get vel_n+1/2 */
        GVG_rk_average_flow_variables( u, v, w, c, params, tvd_coef1, tvd_coef2);

#ifdef TVD_RK3

        /*****************************************/
        /* 3rd RK step, advance time to t+1.5dt */

        /*find convective outflow boundary values, at time t + 1.5*dt*/
        if (params->outflow) {

            Outflow_impose_convective_boundary( u, v, w, c, grid, params, dt) ;
        }

        GVG_integrate_all_the_equations_in_time(u, v, w, p, c, grid, ENO, params, dt, dt_old, 3);

        /* Now have vel_n+1/2 and vel_n+3/2, average to get vel_n+1 */

        GVG_rk_average_flow_variables(u, v, w, c, params, 1.0/3.0, 2.0/3.0);

#endif
        dt_old = dt;
        time  += dt_old;

        //Conc_check_valid(c);

        /* Integrate (in time) deposited height (from particles */
        if (params->deposit_height_output) {

            for (iconc=0; iconc<NConc; iconc++) {

                if (c[iconc]->Type == PARTICLE) {

                    Conc_integrate_deposited_height(c[iconc], grid, params->Conc_alpha[iconc], dt_old);
                } /* if */
            } /* for iconc */
        } /* if */

        if ( (params->dump_conc) && (iter % 20 == 0) && (time > 1.0) ){

            /* Write the final deposit profile */
            PetscPrintf(PCW, "gvg.c/ Writing dumped conc to file at time:%f (warning the file index may not be accurate) \n", time);
            Conc_dump_particles(c, grid, params);
            Output_write_ascii_deposit_height_dumped(output, c, grid, params, output_iter) ;

        } /* if dump_conc */

        iter++;

        PetscPrintf(PCW, "    ***************************************************     \n");
        PetscPrintf(PCW, "    iteration %d, dt = %.4lf, dt_old = %f time = %.3lf\n", iter, dt, dt_old, time);
        PetscPrintf(PCW, "    ***************************************************    \n");

#ifdef GET_TIMES
        if (iter % 50 == 0) {

            GVG_announce_times(time);
        }
#endif
    } /* main while */

#ifdef MEMORY_PROFILING
    Display_memory_info();
#endif

#ifdef GET_TIMES
    ierr = PetscGetTime(&T_End); PETScErrAct(ierr);
    PetscPrintf(PCW, "Total time elapsed on processor zero was:%f\n", (T_End-T_Start));
    PetscPrintf(PCW, "Total time elapsed on processor zero for solution of Poisson equation was:%f\n", W_total_Poisson_solver_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for solution of uvwc equations was:%f\n", W_total_uvwc_solution_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for writing was:%f\n", W_total_writing_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for communication was:%f\n", W_total_external_comm_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for rhs (and extra work) was:%f\n", W_total_rhs_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for convective terms was:%f\n", W_total_convective_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for ENO calculations was:%f\n", W_total_ENO_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for ENO setup was:%f\n",	W_total_convective_setup_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for ENO setup was:%f\n", W_total_ENO_setup_time);
    PetscPrintf(PCW, "Total time elapsed on processor zero for ENO deposition was:%f\n", W_total_ENO_deposition_time);
#endif
    /* Destroy ENO variables */
    ENO_destroy(ENO);

    /* Destroy output writer */
    Output_destroy(output);
    return iter;

}
/***************************************************************************************************/

/* This function returns the new time step using CFL condition */
/* Also, it computes the new dt in such a way that next saving time step is sharply exact! */
double GVG_new_time_step(Velocity *u, Velocity *v, Velocity *w, MAC_grid *grid, Parameters *params, double current_time, double next_output_time, double dt_inp) {

    double dt_test, dt_out;
    double next_time;
    double next_neighboring_output_time;

    /* First, find dt based on the CFL condition (convective terms) */
    dt_test = GVG_compute_dt_applying_cfl_condition(u, v, w, grid, params);

    /* This is done to avoid sharp jump in time steps. */
    if ( dt_test > (2.0 * dt_inp) ) {

        dt_out = 2.0*dt_inp;
    }
    else {

        dt_out = dt_test;
    }

    /* Now check if the time is greater than the saving time. If yes, find the suitable dt that which
 saves the data at the exact time */

    next_time = current_time + dt_out;
    next_neighboring_output_time = next_output_time - 0.5*dt_out;

    /* This is done to avoid very small values of dt which could slow down the simulation very significantly */
    if ( ( next_time > next_neighboring_output_time) && ( next_time <= next_output_time) ) {

        dt_out *= 0.5;
    } else {
        if ( next_time > next_output_time ) {
            dt_out = (next_output_time - current_time);
	}
    }
	//PetscPrintf(PCW, "gvg.c/ dt_test:%f dt_out:%f dt_inp:%f time:%f next_n_time:%f next_output_time:%f \n", dt_test, dt_out, dt_inp, current_time, next_neighboring_output_time, next_output_time);
    return (dt_out);
}
/***************************************************************************************************/

/* This function computes Delta_t by applying CFL condition */
double GVG_compute_dt_applying_cfl_condition(Velocity *u, Velocity *v, Velocity *w, MAC_grid *grid, Parameters *params) {

    double a_dx, a_dy, a_dz;
    double max_a_dt = 0.0;
    double test_a_dt;
    int i, j, k;
    int NX, NY, NZ;
    double ***u_data_bc, ***v_data_bc, ***w_data_bc;
    double *xu, *yv, *zw;
    double dt=0.0;
    double alpha;
    double u_cfl, v_cfl, w_cfl, u_cell, v_cell, w_cell;
    short int cfl_method;
    int ierr;
    int Is, Js, Ks;
    int Ie, Je, Ke;
    double W_dt_min;
    double u_max  = 0.0;
    double W_u_max;
    double send_data[2];
    double W_min_data[2];


    cfl_method = params->cfl_method; /*Default 2 */

    /* First, find the velocities at the cell center */
    //Velocity_cell_center(u, v, w, grid, params);

    NX  = grid->NX;
    NY  = grid->NY;
    NZ  = grid->NZ;

    /* get the global array for Velocity at cell center */
    ierr = DAVecGetArray(grid->DA_3D, u->G_data_bc, (void ***)&u_data_bc);PETScErrAct(ierr);
    ierr = DAVecGetArray(grid->DA_3D, v->G_data_bc, (void ***)&v_data_bc);PETScErrAct(ierr);
    ierr = DAVecGetArray(grid->DA_3D, w->G_data_bc, (void ***)&w_data_bc);PETScErrAct(ierr);

    xu = grid->xu;
    yv = grid->yv;
    zw = grid->zw;

    alpha = params->alpha; /* dt_ = dt_cfl * alpha */

    /* Start index of bottom-left-back corner on current processor */
    Is = grid->G_Is;
    Js = grid->G_Js;
    Ks = grid->G_Ks;

    /* End index of top-right-front corner on current processor */
    Ie = grid->G_Ie;
    Je = grid->G_Je;
    Ke = grid->G_Ke;


    if (cfl_method == 1) {

        /* Use other methods */
    }
    else if (cfl_method == 2) {

        double v_particle;
        double settling_speed_max = 0.0;
        int iconc;
        int NConc = params->NConc;


        /* find the maximum particle settling speed */
        for (iconc=0; iconc<NConc; iconc++) {

            if (fabs(params->V_s0[iconc]) > settling_speed_max ) {

                settling_speed_max = fabs(params->V_s0[iconc]);
            } /* if */
        } /* for */


        /*check out every cell in bc coordinates*/
        for (k=Ks; k<Ke; k++) {
            for (j=Js; j<Je; j++) {
                for (i=Is; i<Ie; i++) {

                    if ( (Grid_get_c_status(grid, i, j, k) == FLUID) || (Grid_get_c_status(grid, i, j, k) == IMMERSED) ){

                        a_dz = 1.0/(zw[k+1] - zw[k]);
                        a_dy = 1.0/(yv[j+1] - yv[j]);
                        a_dx = 1.0/(xu[i+1] - xu[i]);

                        u_cell     = fabs(u_data_bc[k][j][i]);
                        v_cell     = fabs(v_data_bc[k][j][i]);
                        w_cell     = fabs(w_data_bc[k][j][i]);
                        v_particle = fabs(v_data_bc[k][j][i] - settling_speed_max);

                        u_cfl = u_cell;
                        v_cfl = max(v_cell, v_particle);
                        w_cfl = w_cell;

                        /* To find maximum u-Velocity in the domain */
                        if (u_cell > u_max) {

                            u_max = u_cell;
                        }

                        test_a_dt = u_cfl * a_dx + v_cfl * a_dy + w_cfl * a_dz;
                        if (test_a_dt > max_a_dt) {

                            max_a_dt = test_a_dt;
                        }
                    } /* if !solid */
                } /* for i*/
            } /* for j*/
        } /* for k*/

        if (max_a_dt <= 1e-8) {

            dt = 0.4; /* big delta_t */
        } else {

            dt = alpha / max_a_dt;
        }

    } /* if */

    /* Now, find minimum dt and send it back to all processors */
    /* Pack data in one array and send to all processors to find dt_min and 1/u_max */
    send_data[0] = dt;
    send_data[1] = 1.0/u_max; /* inverse u_max to send it with "dt" */

    (void)MPI_Allreduce ( (void *)send_data, (void *)W_min_data, 2, MPI_DOUBLE, MPI_MIN, PCW);


    /* Global minimum dt */
    W_dt_min = W_min_data[0];

    /* Update the velocity for the convective outflow boundary condition */
    if ( (params->outflow) && (params->outflow_type == CONVECTIVE_MAXIMUM_VELOCITY) ) {

        /* Global maximum u-velocity */
        W_u_max   = 1.0/W_min_data[1];
        params->U = 1.10*W_u_max;
    } /* if */

    /* Check if the "dt" for the convective outflow restricts the dt for the other parts of the domain */
    //if (params->outflow) {

    //dx = xu[NX-1] - xu[NX-2];

    //dt_outflow = 1.0 * dx / params->U; /* CFL) max = 1.0 */
    //if (dt_outflow < dt) {

    //return (dt_outflow);
    //}
    //}

    ierr = DAVecRestoreArray(grid->DA_3D, u->G_data_bc, (void ***)&u_data_bc);PETScErrAct(ierr);
    ierr = DAVecRestoreArray(grid->DA_3D, v->G_data_bc, (void ***)&v_data_bc);PETScErrAct(ierr);
    ierr = DAVecRestoreArray(grid->DA_3D, w->G_data_bc, (void ***)&w_data_bc);PETScErrAct(ierr);

    return (W_dt_min);
}
/***************************************************************************************************/

/* This function does the RK averaging for u,v and conc. */
void GVG_rk_average_flow_variables(Velocity *u, Velocity *v, Velocity *w, Concentration **c, Parameters *params, double old_frac, double new_frac) {

    int iconc, NConc;

    NConc = params->NConc;

    Velocity_rk_average(u, old_frac, new_frac);
    Velocity_rk_average(v, old_frac, new_frac);
    Velocity_rk_average(w, old_frac, new_frac);

    for (iconc=0; iconc<NConc; iconc++) {

        Conc_rk_average(c[iconc], old_frac, new_frac);
    }

    if (params->outflow) {

        Velocity_rk_average_outflow(u, old_frac, new_frac);
        Velocity_rk_average_outflow(v, old_frac, new_frac);
        Velocity_rk_average_outflow(w, old_frac, new_frac);
        for (iconc=0; iconc<NConc; iconc++) {

            Conc_rk_average_outflow(c[iconc], old_frac, new_frac);
        }
    }

}
/***************************************************************************************************/

/* This function copies the "data" property of u, v and conc into data_old for time integration purposes. */
void GVG_store_flow_variables_old_data(Velocity *u, Velocity *v, Velocity *w, Concentration **c, Parameters *params) {

    int NConc, iconc;

    NConc = params->NConc;

    Velocity_store_old_data(u);
    Velocity_store_old_data(v);
    Velocity_store_old_data(w);

    for (iconc=0; iconc<NConc; iconc++) {

        Conc_store_old_data(c[iconc]);
    } /* iconc */

    if (params->outflow) {

        Velocity_store_old_outflow(u);
        Velocity_store_old_outflow(v);
        Velocity_store_old_outflow(w);

        for (iconc=0; iconc<NConc; iconc++) {

            Conc_store_old_outflow(c[iconc]);
        } /* for iconc */
    } /* if outflow */

}
/***************************************************************************************************/

/* This function writes the essential workspace data to binary files. Backup for future simualtions */
void GVG_write_workspace_data(GVG_bag *data_bag, Resume *resume) {

    int NConc, iconc;
    int set_data_flag;

    NConc = data_bag->params->NConc;

    /* Just write data */
    Resume_set_write_flag(resume);
    /* Do not read data */
    Resume_reset_read_flag(resume);

    if (data_bag->params->constant_geometry) {

        /* Do not write grid geometry */
        Resume_reset_IO_grid_flag(resume);
    } else {

        /* Save grid cell center geometry since it changes in time */
        /* Do not use it now. It is not complete yet */
        Resume_set_IO_grid_flag(resume);
    }

    /* Copy the outflow data into the last plane of the 3d data array, i.e. G_data of velocity and concentration fields */
    if (data_bag->params->outflow) {

        set_data_flag = YES;
        Outflow_vel_copy_data_into_last_plane(data_bag->u, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->v, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->w, data_bag->grid, set_data_flag);

        for(iconc=0; iconc<NConc; iconc++) {

            Outflow_concentration_copy_data_into_last_plane(data_bag->c[iconc], data_bag->grid, set_data_flag);
        }
    } /* if outflow */

    /* Set the pointers */
    Resume_set_data(resume, data_bag);

    /* Write primary data, i.e. dt, dt_old, iter, time, .... */
    Resume_write_primary_data(resume, data_bag);

    /* Write data for parallel vectors in binary */
    Resume_write_data(resume);

    /* For now, reset the data after being written to file. Set them back to zero, since they are stored in
 separate arrays, i.e. G_outfloe[][] */
    if (data_bag->params->outflow) {

        set_data_flag = NO;
        Outflow_vel_copy_data_into_last_plane(data_bag->u, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->v, data_bag->grid, set_data_flag);
        Outflow_vel_copy_data_into_last_plane(data_bag->w, data_bag->grid, set_data_flag);

        for(iconc=0; iconc<NConc; iconc++) {

            Outflow_concentration_copy_data_into_last_plane(data_bag->c[iconc], data_bag->grid, set_data_flag);
        } /* for iconc */
    } /* if outflow */

    if (data_bag->params->sedimentation) {

        char filename[FILENAME_MAX];
        for(iconc=0; iconc<NConc; iconc++) {

            if (data_bag->c[iconc]->Type == PARTICLE) {

                sprintf(filename, "Resume_sed_height_c%d_data.bin", iconc);
                int NZ = data_bag->grid->NZ;
                int NX = data_bag->grid->NX;
                double **data = NULL;

                data = data_bag->c[iconc]->W_deposit_height;

                Resume_write_2D_data(resume, data, filename, NX, NZ);
            } /* if */
        } /* for iconc */
    }
}
/***************************************************************************************************/

/* This function defines the geometry and then tags the nodes based on the location of the node with respect to the solid interface as */
/* SOLID, FLUID, IMMERSED, BOUNDARY and OUTSIDE */
void GVG_identify_geometry(MAC_grid *grid, Parameters *params) { 

    /* First, generate the location of the interface at the grid centers */
    Grid_describe_interface(grid, params);
    PetscPrintf(PCW, "gvg.c/ Grid has been described successfully\n");

    /* Initialize the surface using either the cell-center values of the interface or any custom levelset function */
    /* In order to define multiple solid objects, digg into the function below and define the correct associated levelset function with those */
    //grid->surf = Surface_create(grid);


    Surface_initialize(grid->surf, grid, params);
    PetscPrintf(PCW, "gvg.c/ surface has been intialized successfully\n");

    /* Iterate in fictitious time to find the exact location and signed distance functions of all the other nodes in the domain with respect to the exact location of the interface */
    Surface_find_exact_sdf(grid->surf, grid);

    Surface_find_q_sdf(grid->surf, grid);
    PetscPrintf(PCW, "gvg.c/ Surface sdf intialized successfully\n");

    //Display_3D_variable(grid->surf->c_sdf, grid->NX, grid->NY, grid->NZ, "c sdf");
    //getchar();
    //Display_3D_variable(grid->surf->u_sdf, grid->NX, grid->NY, grid->NZ, "u sdf");
    //getchar();
    //Display_3D_variable(grid->surf->v_sdf, grid->NX, grid->NY, grid->NZ, "v sdf");
    //getchar();
    //Display_3D_variable(grid->surf->w_sdf, grid->NX, grid->NY, grid->NZ, "w sdf");
    //getchar();

    /* This function tags all the nodes (u, v, w, c) based on the surface position. */
    /* Tagging is done via a levelset represenation of the interface for the solid inteface */
    Grid_identify_geometry(grid);
    PetscPrintf(PCW, "gvg.c/ Surface sdf intialized successfully\n");

    PetscPrintf(PCW, "gvg.c/ Geometry has been created successfully \n");
}


/* This function compute Convective term. i.e. udqdx + vdqdy  using ENO scheme. Note that q could be any flow
varibale
q (which_quantity) :
 - u
 - v
 - w
 - c
*/
void GVG_compute_convective_terms(ENO_Scheme *ENO, Velocity *u, Velocity *v, Velocity *w, Concentration *c, MAC_grid *grid, Parameters *params, char which_quantity) {

    int m;
    int local_index, Nmax_local;
    int i_index, j_index, k_index;
    Indices start_cell_indices, end_cell_indices;
    int ierr ;
    int fluid_index, ex;

    /* Same for all quantities */
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    /* Start index of bottom-left-back corner on current processor */
    int Is = grid->G_Is;
    int Js = grid->G_Js;
    int Ks = grid->G_Ks;

    /* End index of top-right-front corner on current processor */
    int Ie = grid->G_Ie;
    int Je = grid->G_Je;
    int Ke = grid->G_Ke;

    /* Start index of bottom-left-back corner on current processor including ghost nodes */
    int Is_g = grid->L_Is;
    int Js_g = grid->L_Js;
    int Ks_g = grid->L_Ks;

    /* End index of top-right-front corner on current processor including ghost nodes */
    int Ie_g = grid->L_Ie;
    int Je_g = grid->L_Je;
    int Ke_g = grid->L_Ke;


    double ***u_data=NULL;
    double ***v_data=NULL;
    double ***w_data=NULL;
    double ***q=NULL;
    double ***conv=NULL;
    double *xq=NULL;
    double *yq=NULL;
    double *zq=NULL;

    int i_start=-1;
    int i_end=-1;
    int j_start=-1;
    int j_end=-1;
    int k_start=-1;
    int k_end=-1;
    switch (which_quantity) {

    case 'u':

        /* Get the local velocities */
        ierr = DAVecGetArray(grid->DA_3D, u->L_data, (void ***)&u_data); PETScErrAct(ierr);
        /* other velocities at u-grid */
        ierr = DAVecGetArray(grid->DA_3D, *u->L_v_transposed, (void ***)&v_data); PETScErrAct(ierr);
        ierr = DAVecGetArray(grid->DA_3D, *u->L_w_transposed, (void ***)&w_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecGetArray(grid->DA_3D, u->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms: uddx+vddy+wddz */
        ierr = VecSet(u->G_conv, 0.0);
        ierr = DAVecGetArray(grid->DA_3D, u->G_conv, (void ***)&conv); PETScErrAct(ierr);

        /* grid */
        xq = grid->xu;
        yq = grid->yu;
        zq = grid->zu;

        /* indices start and end on current processor */
        i_start = max(1,Is); /* i=0, i=NX-1 are not included */
        j_start = Js;
        k_start = Ks;

        /* exclude the half cell added */
        i_end   = min(NX-1, Ie);
        j_end   = min(NY-1, Je);
        k_end   = min(NZ-1, Ke);

        break;

    case 'v':

        /* Get the local velocities */
        ierr = DAVecGetArray(grid->DA_3D, v->L_data, (void ***)&v_data); PETScErrAct(ierr);
        /* other velocities at v-grid */
        ierr = DAVecGetArray(grid->DA_3D, *v->L_u_transposed, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecGetArray(grid->DA_3D, *v->L_w_transposed, (void ***)&w_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecGetArray(grid->DA_3D, v->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms: uddx+vddy+wddz */
        ierr = VecSet(v->G_conv, 0.0);
        ierr = DAVecGetArray(grid->DA_3D, v->G_conv, (void ***)&conv); PETScErrAct(ierr);

        xq = grid->xv;
        yq = grid->yv;
        zq = grid->zv;

        /* indices start and end on current processor */
        i_start = Is;
        j_start = max(1, Js);/* j=0, j=NY-1 are not included */
        k_start = Ks;

        /* exclude the half cell added */
        i_end   = min(NX-1, Ie);
        j_end   = min(NY-1, Je);
        k_end   = min(NZ-1, Ke);

        break;

    case 'w':
        /* Get the local velocities */
        ierr = DAVecGetArray(grid->DA_3D, w->L_data, (void ***)&w_data); PETScErrAct(ierr);
        /* other velocities at w-grid */
        ierr = DAVecGetArray(grid->DA_3D, *w->L_u_transposed, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecGetArray(grid->DA_3D, *w->L_v_transposed, (void ***)&v_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecGetArray(grid->DA_3D, w->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms: uddx+vddy+wddz */
        ierr = VecSet(w->G_conv, 0.0);
        ierr = DAVecGetArray(grid->DA_3D, w->G_conv, (void ***)&conv); PETScErrAct(ierr);

        /* grid location */
        xq = grid->xw;
        yq = grid->yw;
        zq = grid->zw;

        /* indices start and end on current processor */
        i_start = Is; /* k=0, k=NZ-1 are not included */
        j_start = Js;
        k_start = max(1, Ks);

        /* exclude the half cell added */
        i_end   = min(NX-1, Ie);
        j_end   = min(NY-1, Je);
        k_end   = min(NZ-1, Ke);

        break;

    case 'c':

        /* Get the local velocities. Cell center velocities */
        ierr = DAVecGetArray(grid->DA_3D, u->L_data_bc, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecGetArray(grid->DA_3D, w->L_data_bc, (void ***)&w_data); PETScErrAct(ierr);

        if (c->Type == PARTICLE) { /* v_particle = fluid velocity + setteling speed */

            ierr = DAVecGetArray(grid->DA_3D, c->L_v_particle, (void ***)&v_data); PETScErrAct(ierr);
        } else { /* fluid velocity */

            ierr = DAVecGetArray(grid->DA_3D, v->L_data_bc, (void ***)&v_data); PETScErrAct(ierr);
        } /* else */

        /* quantity */
        ierr = DAVecGetArray(grid->DA_3D, c->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms: uddx+vddy+wddz */
        ierr = VecSet(c->G_conv, 0.0);
        ierr = DAVecGetArray(grid->DA_3D, c->G_conv, (void ***)&conv); PETScErrAct(ierr);

        xq = grid->xc;
        yq = grid->yc;
        zq = grid->zc;

        /* indices start and end on current processor */
        i_start = Is;
        j_start = Js;
        k_start = Ks;

        /* exclude the half cell added */
        i_end   = min(NX-1, Ie);
        j_end   = min(NY-1, Je);
        k_end   = min(NZ-1, Ke);


        break;
    } /* swicth */

    double *D0         = ENO->D0;
    double *Position   = ENO->Position;
    double *Velocity   = ENO->Velocity;
    double *VdqdX      = ENO->VdQdX;
    int ENO_order  = ENO->ENO_order;

    /******************************************************/
    int i, j, k;

    /*get udqdx, do row by row*/
    for (k=k_start; k<k_end; k++) {
        for (j=j_start; j<j_end; j++) {

            /*set up zeroth divided difference*/
            /* copy all the data into the arrays */
            local_index      = ENO_order;

            /* indices of the first and last nodes of the data to be copied into ENO scheme */
            start_cell_indices.x_index = i_start;
            start_cell_indices.y_index = j;
            start_cell_indices.z_index = k;

            end_cell_indices.x_index = i_end-1;
            end_cell_indices.y_index = j;
            end_cell_indices.z_index = k;


            /* go through local nodes + ghost nodes from neighboring processors */
            for (i=i_start; i<i_end; i++) {

                /* Current quantity node is a fluid node */

                /* Copy the value of current node into the D0 array */
                D0[local_index]       = q[k][j][i];
                Velocity[local_index] = u_data[k][j][i];
                Position[local_index] = xq[i];

                local_index++;
            } /* for i */

            /* Now, check if the current processor is NOT in the vicinity of the two end boundaries, copy the value of the ghost nodes into the data arrays */
            /* Is == Is_g == 0 */
            if (Is_g != Is) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ex;
                    fluid_index = Is_g + ex;
                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[k][j][fluid_index];
                    Velocity[local_index] = u_data[k][j][fluid_index];
                    Position[local_index] = xq[fluid_index];

                } /* for ex */
            } /* if */

            /* Ie == Ie_g == NX */
            if (Ie_g != Ie) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ENO_order + (i_end - i_start) + ex;
                    fluid_index = i_end + ex;

                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[k][j][fluid_index];
                    Velocity[local_index] = u_data[k][j][fluid_index];
                    Position[local_index] = xq[fluid_index];

                } /* for ex */
            } /* if */


            /* Now, generate the ghost nodes at both ends of the 1D array of D0 */
            switch (which_quantity) {

            case 'u':
                Velocity_u_ENO_ghost_cells(ENO, u, grid, params, start_cell_indices, end_cell_indices, 'x');
                break;

            case 'v':
                Velocity_v_ENO_ghost_cells(ENO, v, grid, params, start_cell_indices, end_cell_indices, 'x');
                break;

            case 'w':
                Velocity_w_ENO_ghost_cells(ENO, w, grid, params, start_cell_indices, end_cell_indices, 'x');
                break;

            case 'c':
                Conc_ENO_ghost_cells(ENO, c, grid, params, start_cell_indices, end_cell_indices, 'x');
                break;
            } /* switch */

            /* total number of the nodes including the ghost nodes */
            Nmax_local = (i_end - i_start) + 2*ENO_order;

            /* Now, compute the convective term using 3rd order ENO, udqdx */
            if (which_quantity == 'c') {
                ENO_compute_convective_terms(ENO, Nmax_local);
            } else {

                //ENO_compute_convective_terms_central(ENO, Nmax_local);
                ENO_compute_convective_terms(ENO, Nmax_local);
            }

            /* Now, copy the caluculated udqdx back into the data array */
            for (m=ENO_order; m<Nmax_local-ENO_order; m++) {

                /* The real i_index */
                i_index    = (m - ENO_order) + i_start;
                conv[k][j][i_index] += VdqdX[m];


            } /* for m*/
        } /* for j*/
    } /* for k*/

    /******************************************************/
    /******************************************************/
    /*get vdqdy, do column by column in y-direction */
    for (k=k_start; k<k_end; k++) {
        for (i=i_start; i<i_end; i++) {

            local_index      = ENO_order;

            /* indices of the first and last nodes of the data to be copied into ENO scheme */
            start_cell_indices.x_index = i;
            start_cell_indices.y_index = j_start;
            start_cell_indices.z_index = k;

            end_cell_indices.x_index = i;
            end_cell_indices.y_index = j_end-1;
            end_cell_indices.z_index = k;

            for (j=j_start; j<j_end; j++) {


                /* Copy the value of current node into the D0 array */
                D0[local_index]       = q[k][j][i];
                Velocity[local_index] = v_data[k][j][i];
                Position[local_index] = yq[j];

                local_index++;
            } /*for j*/

            /* Now, check if the current processor is NOT in the vicinity of the two end boundaries, copy the value of the ghost nodes into the data arrays */
            /* Js == Js_g == 0 */
            if (Js_g != Js) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ex;
                    fluid_index = Js_g + ex;
                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[k][fluid_index][i];
                    Velocity[local_index] = v_data[k][fluid_index][i];
                    Position[local_index] = yq[fluid_index];

                } /* for ex */
            } /* if */

            /* Je == Je_g == NY */
            if (Je_g != Je) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ENO_order + (j_end - j_start) + ex;
                    fluid_index = j_end + ex;

                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[k][fluid_index][i];
                    Velocity[local_index] = v_data[k][fluid_index][i];
                    Position[local_index] = yq[fluid_index];

                } /* for ex */
            } /* if */


            /* Now, generate the ghost nodes at both ends of the 1D array of D0 */
            switch (which_quantity) {

            case 'u':
                Velocity_u_ENO_ghost_cells(ENO, u, grid, params, start_cell_indices, end_cell_indices, 'y');
                break;

            case 'v':
                Velocity_v_ENO_ghost_cells(ENO, v, grid, params, start_cell_indices, end_cell_indices, 'y');
                break;

            case 'w':
                Velocity_w_ENO_ghost_cells(ENO, w, grid, params, start_cell_indices, end_cell_indices, 'y');
                break;

            case 'c':
                Conc_ENO_ghost_cells(ENO, c, grid, params, start_cell_indices, end_cell_indices, 'y');
                break;
            } /* switch */

            /* total number of the nodes including the ghost nodes */
            Nmax_local = (j_end - j_start) + 2*ENO_order;

            /* Now, compute the convective term using 3rd order ENO, udqdx */
            if (which_quantity == 'c') {

                ENO_compute_convective_terms(ENO, Nmax_local);
            } else {

                //ENO_compute_convective_terms_central(ENO, Nmax_local);
                ENO_compute_convective_terms(ENO, Nmax_local);
            }

            /* Now, copy the caluculated udcdx back into the data array */
            for (m=ENO_order; m<Nmax_local-ENO_order; m++) {

                /* The real i_index */
                j_index    = (m - ENO_order) + j_start;
                conv[k][j_index][i] += VdqdX[m];
            } /* for m*/

        } /* for i*/
    } /* for k*/

    /******************************************************/
    /******************************************************/
    /*get wdqdz, do column by column in y-direction */
    for (j=j_start; j<j_end; j++) {
        for (i=i_start; i<i_end; i++) {

            /* indices of the first and last nodes of the data to be copied into ENO scheme */
            start_cell_indices.x_index = i;
            start_cell_indices.y_index = j;
            start_cell_indices.z_index = k_start;

            end_cell_indices.x_index = i;
            end_cell_indices.y_index = j;
            end_cell_indices.z_index = k_end-1;

            local_index      = ENO_order;
            for (k=k_start; k<k_end; k++) {


                /* Copy the value of current node into the D0 array */
                D0[local_index]       = q[k][j][i];
                Velocity[local_index] = w_data[k][j][i];
                Position[local_index] = zq[k];

                local_index++;

            } /* for k */

            /* Now, check if the current processor is NOT in the vicinity of the two end boundaries, copy the value of the ghost nodes into the data arrays */
            /* Ks == Ks_g == 0 */
            if (Ks_g != Ks) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ex;
                    fluid_index = Ks_g + ex;
                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[fluid_index][j][i];
                    Velocity[local_index] = w_data[fluid_index][j][i];
                    Position[local_index] = zq[fluid_index];

                } /* for ex */
            } /* if */

            /* Ke == Ke_g == NK */
            if (Ke_g != Ke) {

                for (ex=0; ex<ENO_order; ex++) {

                    local_index = ENO_order + (k_end - k_start) + ex;
                    fluid_index = k_end + ex;

                    /* Copy the value of current node into the D0 array */
                    D0[local_index]       = q[fluid_index][j][i];
                    Velocity[local_index] = w_data[fluid_index][j][i];
                    Position[local_index] = zq[fluid_index];

                } /* for ex */
            } /* if */

            /* Now, generate the ghost nodes at both ends of the 1D array of D0 */
            switch (which_quantity) {

            case 'u':
                Velocity_u_ENO_ghost_cells(ENO, u, grid, params, start_cell_indices, end_cell_indices, 'z');
                break;

            case 'v':
                Velocity_v_ENO_ghost_cells(ENO, v, grid, params, start_cell_indices, end_cell_indices, 'z');
                break;

            case 'w':
                Velocity_w_ENO_ghost_cells(ENO, w, grid, params, start_cell_indices, end_cell_indices, 'z');
                break;

            case 'c':
                Conc_ENO_ghost_cells(ENO, c, grid, params, start_cell_indices, end_cell_indices, 'z');
                break;
            } /* switch */


            /* total number of the nodes including the ghost nodes */
            Nmax_local = (k_end - k_start) + 2*ENO_order;

            /* Now, compute the convective term using 3rd order ENO, udqdx */
            if (which_quantity == 'c') {
                ENO_compute_convective_terms(ENO, Nmax_local);
            } else {

                //ENO_compute_convective_terms_central(ENO, Nmax_local);
                ENO_compute_convective_terms(ENO, Nmax_local);
            }

            /* Now, copy the caluculated udcdx back into the data array */
            for (m=ENO_order; m<Nmax_local-ENO_order; m++) {

                /* The real i_index */
                k_index    = (m - ENO_order) + k_start;

                conv[k_index][j][i] += VdqdX[m];
            } /* for m*/

        } /* for i*/
    } /* for j*/

    /* Now restore arrays */
    switch (which_quantity) {

    case 'u':


        ierr = DAVecRestoreArray(grid->DA_3D, u->L_data, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *u->L_v_transposed, (void ***)&v_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *u->L_w_transposed, (void ***)&w_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecRestoreArray(grid->DA_3D, u->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms */
        ierr = DAVecRestoreArray(grid->DA_3D, u->G_conv, (void ***)&conv); PETScErrAct(ierr);

        break;

    case 'v':

        ierr = DAVecRestoreArray(grid->DA_3D, v->L_data, (void ***)&v_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *v->L_u_transposed, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *v->L_w_transposed, (void ***)&w_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecRestoreArray(grid->DA_3D, v->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms */
        ierr = DAVecRestoreArray(grid->DA_3D, v->G_conv, (void ***)&conv); PETScErrAct(ierr);

        break;

    case 'w':

        ierr = DAVecRestoreArray(grid->DA_3D, w->L_data, (void ***)&w_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *w->L_u_transposed, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, *w->L_v_transposed, (void ***)&v_data); PETScErrAct(ierr);

        /* quantity */
        ierr = DAVecRestoreArray(grid->DA_3D, w->L_data, (void ***)&q); PETScErrAct(ierr);


        /* Convective terms */
        ierr = DAVecRestoreArray(grid->DA_3D, w->G_conv, (void ***)&conv); PETScErrAct(ierr);

        break;

    case 'c':

        /* Get the local velocities. Cell center velocities */
        ierr = DAVecRestoreArray(grid->DA_3D, u->L_data_bc, (void ***)&u_data); PETScErrAct(ierr);
        ierr = DAVecRestoreArray(grid->DA_3D, w->L_data_bc, (void ***)&w_data); PETScErrAct(ierr);

        if (c->Type == PARTICLE) { /* v_particle = fluid velocity + setteling speed */

            ierr = DAVecRestoreArray(grid->DA_3D, c->L_v_particle, (void ***)&v_data); PETScErrAct(ierr);
        } else { /* fluid velocity */

            ierr = DAVecRestoreArray(grid->DA_3D, v->L_data_bc, (void ***)&v_data); PETScErrAct(ierr);
        }

        /* quantity */
        ierr = DAVecRestoreArray(grid->DA_3D, c->L_data, (void ***)&q); PETScErrAct(ierr);

        /* Convective terms */
        ierr = DAVecRestoreArray(grid->DA_3D, c->G_conv, (void ***)&conv); PETScErrAct(ierr);

        break;
    } /* swicth */

}
/***************************************************************************************************/

/* This function integrate u,v and conc transport equation for one time step (First order in Time) */
void GVG_integrate_all_the_equations_in_time(Velocity *u, Velocity *v, Velocity *w, Pressure *p, Concentration **c, MAC_grid *grid, ENO_Scheme *ENO, Parameters *params, double dt, double dt_old, short int which_stage) {

    int NConc;
    int iconc;
    int ierr;
    PetscLogDouble T1, T2;

    NConc = params->NConc;

    /* Before hand, velocity at cell center is calculated in the gvg_compute_dt_applying_cfl_condition */
#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
    if (which_stage > 1) {

        /* cell centered velocity: Communication is done for local ghost nodes within the function*/
        Velocity_cell_center(u, v, w, grid, params);
    } /* if */

    /* find other component of velocity on current grid point */

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
    if (params->sedimentation) {

        for (iconc=0; iconc<NConc; iconc++) {



            if (params->hindered_settling) { /* Find particle settling speed based on conc. and particle volume fraction */

            }

            /* find particle v-velocity = v_fluid + v_settling */
            Conc_set_particle_v_velocity(c[iconc], v, params) ;
        } /* for */
    } /* if */

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Communication_update_ghost_nodes(&grid->DA_3D, &u->G_data_bc, &u->L_data_bc, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &v->G_data_bc, &v->L_data_bc, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &w->G_data_bc, &w->L_data_bc, 'I');

    for (iconc=0; iconc<NConc; iconc++) {

        Communication_update_ghost_nodes(&grid->DA_3D, &c[iconc]->G_data, &c[iconc]->L_data, 'I');
        if (c[iconc]->Type == PARTICLE) {

            Communication_update_ghost_nodes(&grid->DA_3D, &c[iconc]->G_v_particle, &c[iconc]->L_v_particle, 'I');
        } /* if */
    } /* for iconc */
    /******************************************/

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_external_comm_time += T2 - T1;
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    /* find all convective derivatives using 3rd order ENO scheme*/
    /* This order is important. Keep it */
    Velocity_transpose_velocities(u, v, w, grid, params, 'u');
    Communication_update_ghost_nodes(&grid->DA_3D, u->G_v_transposed, u->L_v_transposed, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, u->G_w_transposed, u->L_w_transposed, 'I');
    GVG_compute_convective_terms(ENO, u, v, w, c[0], grid, params, 'u');

    Velocity_transpose_velocities(u, v, w, grid, params, 'v');
    Communication_update_ghost_nodes(&grid->DA_3D, v->G_u_transposed, v->L_u_transposed, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, v->G_w_transposed, v->L_w_transposed, 'I');
    GVG_compute_convective_terms(ENO, u, v, w, c[0], grid, params, 'v');

    Velocity_transpose_velocities(u, v, w, grid, params, 'w');
    Communication_update_ghost_nodes(&grid->DA_3D, w->G_u_transposed, w->L_u_transposed, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, w->G_v_transposed, w->L_v_transposed, 'I');
    GVG_compute_convective_terms(ENO, u, v, w, c[0], grid, params, 'w');

    for (iconc=0; iconc<NConc; iconc++) {

        GVG_compute_convective_terms(ENO, u, v, w, c[iconc], grid, params, 'c');
    } /* for */

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_convective_time += T2 - T1;
#endif

    /* now advance concentration equation*/
    /* Since dt is changing, we have to refind the diagonal part*/
    /* of the LHS matrix. All other coeff. are constant for uniform mesh*/
    if ( which_stage == 1 ) {

        for (iconc=0; iconc<NConc; iconc++) {

            Conc_modify_diagonal(c[iconc], grid, dt, dt_old);
        }
    } /* if which_stage */

    for (iconc=0; iconc<NConc; iconc++) {

#ifdef GET_TIMES
        ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
        Conc_set_RHS(c[iconc], grid, params, dt);

#ifdef GET_TIMES
        ierr = PetscGetTime(&T2);PETScErrAct(ierr);
        W_total_rhs_time += T2 - T1;
#endif

#ifdef GET_TIMES
        ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
        Conc_solve(c[iconc]);

#ifdef GET_TIMES
        ierr = PetscGetTime(&T2);PETScErrAct(ierr);
        W_total_uvwc_solution_time += T2 - T1;
#endif

    } /* for iconc */
    /* compute total concentration field data in presence of more than one concentraion field
  This is to be used in v-momentum equation */
    if (NConc > 1) {

        Conc_compute_total_concentration(c, params);
    } /* if */
    /* LHS matrix changes only if viscosity is changing [ignoring the temporal change */

    /* Update the diagonal part of LHS matrix using dt */
    if ( which_stage == 1 ) {

        Velocity_modify_diagonal(u, grid, params, dt, dt_old);
        Velocity_modify_diagonal(v, grid, params, dt, dt_old);
        Velocity_modify_diagonal(w, grid, params, dt, dt_old);

    } /* if */

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_u_set_RHS(u, p, grid, params, dt);
#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_solve(u);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_uvwc_solution_time += T2 - T1;
#endif

#ifdef SERIOUS_DEBUGGING

    Display_DA_3D_data(u->G_b, grid, params, "rhs u", 'u');
    Display_DA_3D_data(u->G_data, grid, params, "solution u", 'u');
    getchar();

    has_nan = Debugger_check_nan(u->G_data, grid, params, "u-solution");
    printf("gvg.c/ u-solution owns nan(1:yes, 0:no) :%d\n", has_nan);
    has_nan = Debugger_check_nan(u->G_b, grid, params, "u-rhs");
    printf("gvg.c/ u-rhs owns nan(1:yes, 0:no) :%d\n", has_nan);
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_v_set_RHS(v, p, c, grid, params, dt);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_solve(v);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_uvwc_solution_time += T2 - T1;
#endif

#ifdef SERIOUS_DEBUGGING

    Display_DA_3D_data(v->G_b, grid, params, "rhs v", 'v');
    Display_DA_3D_data(v->G_data, grid, params, "solution v", 'v');
    getchar();
    has_nan = Debugger_check_nan(v->G_data, grid, params, "v-solution");
    printf("gvg.c/ v-solution owns nan(1:yes, 0:no) :%d\n", has_nan);
    has_nan = Debugger_check_nan(u->G_b, grid, params, "v-rhs");
    printf("gvg.c/ v-rhs owns nan(1:yes, 0:no) :%d\n", has_nan);
#endif

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_w_set_RHS(w, p, grid, params, dt);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef SERIOUS_DEBUGGING

    Display_DA_3D_data(w->G_b, grid, params, "rhs w", 'w');
    Display_DA_3D_data(w->G_data, grid, params, "solution w", 'w');
    getchar();
#endif

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    Velocity_solve(w);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_uvwc_solution_time += T2 - T1;
#endif

    /* Now, after finding Velocity_star, Communicate to update the ghost nodes from neighboring
 processors. This would be used in pressure equation RHS */
    /******************************************/

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
    Communication_update_ghost_nodes(&grid->DA_3D, &u->G_data, &u->L_data, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &v->G_data, &v->L_data, 'I');
    Communication_update_ghost_nodes(&grid->DA_3D, &w->G_data, &w->L_data, 'I');

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_external_comm_time += T2 - T1;
#endif


    /*Prepare and solve for pressure correction*/
#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
    Pressure_set_RHS(p, u, v, w, grid, params, dt);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif
    Pressure_solve(p);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_Poisson_solver_time += T2 - T1;
#endif

#ifdef GET_TIMES
    ierr = PetscGetTime(&T1);PETScErrAct(ierr);
#endif

    /*use pressure to project a divergence free velocity field and updates the pressure gradients */
    Pressure_project_velocity(p, u, v, w, grid, params, dt);

#ifdef GET_TIMES
    ierr = PetscGetTime(&T2);PETScErrAct(ierr);
    W_total_rhs_time += T2 - T1;
#endif


#ifdef SERIOUS_DEBUGGING_NOW

/*
    GVG_write_PETSC_object_to_file((void *)&u->A, (char *)"U_matrix.dat", PETSC_MAT);
    GVG_write_PETSC_object_to_file((void *)&v->A, (char *)"V_matrix.dat", PETSC_MAT);
    GVG_write_PETSC_object_to_file((void *)&w->A, (char *)"W_matrix.dat", PETSC_MAT);
    GVG_write_PETSC_object_to_file((void *)&p->A, (char *)"P_matrix.dat", PETSC_MAT);
    GVG_write_PETSC_object_to_file((void *)&c[0]->A, (char *)"C_matrix.dat", PETSC_MAT);
*/

    GVG_write_PETSC_object_to_file((void *)&u->G_b, (char *)"U_rhs.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&v->G_b, (char *)"V_rhs.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&w->G_b, (char *)"W_rhs.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&p->G_b, (char *)"P_rhs.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&c[0]->G_b, (char *)"C0_rhs.dat", PETSC_VEC);
	GVG_write_PETSC_object_to_file((void *)&c[1]->G_b, (char *)"C1_rhs.dat", PETSC_VEC);

    GVG_write_PETSC_object_to_file((void *)&u->G_data, (char *)"U_sol.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&v->G_data, (char *)"V_sol.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&w->G_data, (char *)"W_sol.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&p->G_data, (char *)"P_sol.dat", PETSC_VEC);
    GVG_write_PETSC_object_to_file((void *)&c[0]->G_data, (char *)"C_sol.dat", PETSC_VEC);

    printf("gvg.c/ done printing the info\n");
    getchar();
#endif


#ifdef SERIOUS_DEBUGGING
    PetscPrintf(PCW, "-----------u-velocity---------------------------\n");
    PetscPrintf(PCW, "LHS matrix\n");
    //MatView(u->A, 0);
    PetscPrintf(PCW, "**********************RHS*************************: \n");
    VecView(u->G_b, 0);
    PetscPrintf(PCW, "************* u-solution ***********************\n");
    VecView(u->G_data, 0);
    getchar();

    PetscPrintf(PCW, "-----------v-velocity---------------------------\n");
    PetscPrintf(PCW, "LHS matrix\n");
    //MatView(v->A, 0);
    PetscPrintf(PCW, "**********************RHS*************************: \n");
    VecView(v->G_b, 0);
    PetscPrintf(PCW, "************* v-solution ***********************\n");
    VecView(v->G_data, 0);
    getchar();

    PetscPrintf(PCW, "-----------w-velocity---------------------------\n");
    PetscPrintf(PCW, "LHS matrix\n");
    //MatView(w->A, 0);
    PetscPrintf(PCW, "**********************RHS*************************: \n");
    VecView(w->G_b, 0);
    PetscPrintf(PCW, "************* w-solution ***********************\n");
    VecView(w->G_data, 0);
    getchar();

    PetscPrintf(PCW, "-----------Pressure---------------------------\n");
    PetscPrintf(PCW, "LHS matrix\n");
    //MatView(p->A, 0);
    PetscPrintf(PCW, "**********************RHS*************************: \n");
    VecView(p->G_b, 0);
    PetscPrintf(PCW, "************* p-solution ***********************\n");
    VecView(p->G_data, 0);
    getchar();
#endif

#ifdef DEBUGGING_FIND_MAX
    Debugger_q_get_max(u->G_data, grid, params, (char *)"U_max.dat");
    Debugger_q_get_max(v->G_data, grid, params, (char *)"V_max.dat");
    Debugger_q_get_max(w->G_data, grid, params, (char *)"W_max.dat");

    Debugger_q_get_max(c[0]->G_data, grid, params, (char *)"C0_max.dat");
    Debugger_q_get_max(c[1]->G_data, grid, params, (char *)"C1_max.dat");
    Debugger_q_get_max(p->G_data, grid, params, (char *)"p_max.dat");

    Debugger_q_get_max(u->G_b, grid, params, (char *)"U_rhs_max.dat");
    Debugger_q_get_max(v->G_b, grid, params, (char *)"V_rhs_max.dat");
    Debugger_q_get_max(w->G_b, grid, params, (char *)"W_rhs_max.dat");
    Debugger_q_get_max(c[0]->G_b, grid, params, (char *)"C0_rhs_max.dat");
    Debugger_q_get_max(c[1]->G_b, grid, params, (char *)"C1_rhs_max.dat");
    Debugger_q_get_max(p->G_b, grid, params, (char *)"P_rhs_max.dat");
#endif

}
/***************************************************************************************************/
/*
void GVG_printf(int line, const char *file const char *func, const char *format, ...) {

    va_list args;
    int rv;

    int rank;
    int ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); PETScErrAct(ierr);

    //printf("([%d] Fl:%s, Fnc:%s Ln:%d) ", rank, __FILE__, __func__, __LINE__);
    printf("[%d] %s %s-- ", rank, __FILE__, __func__);
    va_start(args, format);
    rv = vprintf(format, args);
    va_end(args);
}
*/

void GVG_set_value(Vec vec_data, MAC_grid *grid) {

    /* Adds constant 'mod' to the diagonal part of matrix A*/
    /* Start index of bottom-left-back corner on current processor */
    int Is = grid->G_Is;
    int Js = grid->G_Js;
    int Ks = grid->G_Ks;

    /* End index of top-right-front corner on current processor */
    int Ie = grid->G_Ie;
    int Je = grid->G_Je;
    int Ke = grid->G_Ke;

    /* Get the diagonal vector array */
    double ***data;
    int ierr = DAVecGetArray(grid->DA_3D, vec_data, (void ***)&data); PETScErrAct(ierr);

    /* Go through the entire domain and only insert the delta_t term for the nodes which are not immersed nodes */
    int i, j, k;
    for (k=Ks; k<Ke; k++) {
        for (j=Js; j<Je; j++) {
            for (i=Is; i<Ie; i++) {

                data[k][j][i] = sin((double)i) + cos((double)j) + cos((double) k);
            }
        }
    }

    ierr = DAVecRestoreArray(grid->DA_3D, vec_data, (void ***)&data); PETScErrAct(ierr);
}


/* This function writes the times for each stage at a certain simulation time */
void GVG_announce_times(double time) {

    int ierr = PetscGetTime(&T_End); PETScErrAct(ierr);
    PetscPrintf(PCW, "Time statistics so far (current simulation time = %f) \n", time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero was:%f\n", (T_End-T_Start));
    PetscPrintf(PCW, "****Total time elapsed on processor zero for solution of Poisson equation was:%f\n", W_total_Poisson_solver_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for solution of uvwc equations was:%f\n", W_total_uvwc_solution_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for writing was:%f\n", W_total_writing_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for communication was:%f\n", W_total_external_comm_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for rhs (and extra work) was:%f\n", W_total_rhs_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for convective terms was:%f\n", W_total_convective_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for ENO calculations was:%f\n", W_total_ENO_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for ENO setup was:%f\n",	W_total_convective_setup_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for ENO setup was:%f\n", W_total_ENO_setup_time);
    PetscPrintf(PCW, "****Total time elapsed on processor zero for ENO deposition was:%f\n", W_total_ENO_deposition_time);

}

void GVG_write_PETSC_object_to_file(void *obj, char *filename, short int what) {

    PetscViewer viewer;

    PetscViewerCreate(PCW, &viewer);
    PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
    PetscViewerFileSetName(viewer, filename);
    if (what == PETSC_MAT) {

        MatView(*(Mat *)obj, viewer);
    } else if (what == PETSC_VEC) {

        VecView(*(Vec *)obj, viewer);
    }

    PetscViewerDestroy(viewer);

}
