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
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

typedef struct { double x,y,z; } dbl3_t;

/* ---------------------------------------------------------------------- */

FixBDOMP::FixBDOMP(LAMMPS *lmp, int narg, char **arg) :
  FixBD(lmp, narg, arg) { }

FixBDOMP::~FixBDOMP() { }

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
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        double randf = sqrt(rmass[i]) * gfactor;
        v[i].x = dtfm * (f[i].x+randf*random->gaussian());
        v[i].y = dtfm * (f[i].y+randf*random->gaussian());
        v[i].z = dtfm * (f[i].z+randf*random->gaussian());
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
        v[i].x = dtfm * (f[i].x+randf*random->gaussian());
        v[i].y = dtfm * (f[i].y+randf*random->gaussian());
        v[i].z = dtfm * (f[i].z+randf*random->gaussian());
        x[i].x += 0.5 * dtv * v[i].x;
        x[i].y += 0.5 * dtv * v[i].y;
        x[i].z += 0.5 * dtv * v[i].z;
      }
  }
}

