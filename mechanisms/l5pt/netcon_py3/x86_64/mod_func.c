#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _ampa_reg(void);
extern void _gsyn_reg(void);
extern void _k2syn_reg(void);
extern void _nmda2_reg(void);
extern void _nmda2_schiller_reg(void);
extern void _pregen_reg(void);
extern void _vecevent_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"ampa.mod\"");
    fprintf(stderr," \"gsyn.mod\"");
    fprintf(stderr," \"k2syn.mod\"");
    fprintf(stderr," \"nmda2.mod\"");
    fprintf(stderr," \"nmda2_schiller.mod\"");
    fprintf(stderr," \"pregen.mod\"");
    fprintf(stderr," \"vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _ampa_reg();
  _gsyn_reg();
  _k2syn_reg();
  _nmda2_reg();
  _nmda2_schiller_reg();
  _pregen_reg();
  _vecevent_reg();
}
