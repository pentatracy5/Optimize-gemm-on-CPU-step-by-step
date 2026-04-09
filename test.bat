@echo off

set PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;%PATH%

set k_start=64
set k_end=4096
set k_blocksize=64

for /L %%k in (%k_start%, %k_blocksize%, %k_end%) do (
    .\x64\Release\BLIS.exe %%k %%k %%k 10
)