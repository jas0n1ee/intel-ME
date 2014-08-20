#!/usr/bin/env python
import sys
f_in=sys.argv[1]
w=int(sys.argv[2])
h=int(sys.argv[3])
l=int(sys.argv[4])
f_out=sys.argv[5]
f_size=w*h*3/2
with open(f_in,'rb') as fd_in, open(f_out,'wb') as fd_out:
	for i in range(15):
		data=fd_in.read(f_size)
		fd_out.write(data)
