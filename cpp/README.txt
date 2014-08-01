

get it running with intel compiler:

  - be sure to have a BOOST version compiled with your intel compiler
    (i.e. not the standard-lib compiled with gcc)
      - download boost
      - unpack it and run ./bootstrap --prefix=[...]
      - compile it (with ./b2)

  - compile with custom boost install:
      cmake .. -DCMAKE_CXX_COMPILER=icpc -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON -DBOOST_INCLUDEDIR=/vol/home/fs172/lib/bmdphi1/include/ -DBOOST_LIBRARYDIR=/vol/home/fs172/lib/bmdphi1/lib/





BUGS:
  - density-based assignment of low-density frames sometimes leaves a single 0-state


