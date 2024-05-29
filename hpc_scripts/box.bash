#!/bin/bash
lftp -c 'open -e "set ftps:initial-prot C"; \
   set ftp:ssl-force true; \
   set ftp:ssl-protect-data true; \
   open ftps://ftp.box.com:990; \
   user dilgrenc@oregonstate.edu; \
   mirror --reverse --delete --no-perms --verbose "/nfs/hpc/share/dilgrenc/topnmusic/model_params_compare" model_params_compare; '