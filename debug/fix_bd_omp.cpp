/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Modified from fix_bd_omp.cpp Mar 6th 2019 version (hsk)
------------------------------------------------------------------------- */

#include "fix_bd_omp.h"
#include "atom.h"
#include "force.h"
#include <math.h>
#include "error.h"
#include "comm.h"
#include "random_mars.h"
#include <omp.h>
#include <string.h>

using namespace LAMMPS_NS;
using namespace FixConst;

typedef struct { double x,y,z; } dbl3_t;

/* ---------------------------------------------------------------------- */

FixBDOMP::FixBDOMP(LAMMPS *lmp, int narg, char **arg) :
  FixBD(lmp, narg, arg) { 
  if (strcmp(style,"bd/omp") != 0 && narg <= 5)
    error->all(FLERR,"Illegal fix bd/omp command");

  t_target = force->numeric(FLERR,arg[3]); // set temperature
  t_period = force->numeric(FLERR,arg[4]); // same as t_period in fix_langevin_overdamp.cpp
  seed = force->inumeric(FLERR,arg[5]); //seed for random number generator. integer

  if (t_target <= 0.0) error->all(FLERR,"Fix bd temperature must be > 0.0");
  if (t_period <= 0.0) error->all(FLERR,"Fix bd period must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix bd command");

  // initialize Marsaglia RNG with processor-unique seed

  #if defined (_OPENMP)
  int nthreads = omp_get_max_threads();
  random_thr = new RanMars*[nthreads];
  for(int tid=0;tid<nthreads;tid++)
    random_thr[tid] = new RanMars(lmp,seed + comm->me);
  #endif

  dynamic_group_allow = 1;
  time_integrate = 1;
}

FixBDOMP::~FixBDOMP() { 
  #if defined (_OPENMP)
  int nthreads = omp_get_num_threads();
  for(int tid=0;tid<nthreads;tid++)
    delete random_thr[tid];
  delete [] random_thr;
  #endif
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixBDOMP::initial_integrate(int /* vflag */)
{
  // update v and x of atoms in group
  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  int i;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        double randf = sqrt(rmass[i]) * gfactor;
        x[i].x += 0.5 * dtv * v[i].x;
        x[i].y += 0.5 * dtv * v[i].y;
        x[i].z += 0.5 * dtv * v[i].z;
      }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        double randf = sqrt(mass[type[i]]) * gfactor;
        x[i].x += 0.5 * dtv * v[i].x;
        x[i].y += 0.5 * dtv * v[i].y;
        x[i].z += 0.5 * dtv * v[i].z;
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBDOMP::final_integrate()
{
  // update v of atoms in group

  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  int i;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++) {
      int tid = omp_get_thread_num();
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        double randf = sqrt(rmass[i]) * gfactor;
        v[i].x = dtfm * (f[i].x+randf*random_thr[tid]->gaussian());
        v[i].y = dtfm * (f[i].y+randf*random_thr[tid]->gaussian());
        v[i].z = dtfm * (f[i].z+randf*random_thr[tid]->gaussian());
        x[i].x += 0.5 * dtv * v[i].x;
        x[i].y += 0.5 * dtv * v[i].y;
        x[i].z += 0.5 * dtv * v[i].z;
      }
    }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++) {
      int tid = omp_get_thread_num();
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        double randf = sqrt(mass[type[i]]) * gfactor;
        v[i].x = dtfm * (f[i].x+randf*random_thr[tid]->gaussian());
        v[i].y = dtfm * (f[i].y+randf*random_thr[tid]->gaussian());
        v[i].z = dtfm * (f[i].z+randf*random_thr[tid]->gaussian());
        x[i].x += 0.5 * dtv * v[i].x;
        x[i].y += 0.5 * dtv * v[i].y;
        x[i].z += 0.5 * dtv * v[i].z;
      }
    }
  }
}

