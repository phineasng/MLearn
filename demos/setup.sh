#!/usr/bin/env bash

set -e

DEMO_ROOT=$(pwd)
BUILD_DIR=${DEMO_ROOT}/demo_build
DEPENDENCIES_DIR=${BUILD_DIR}/dependencies
DEP_INCLUDE_DIR=${DEPENDENCIES_DIR}/include
DEP_LIB_DIR=${DEPENDENCIES_DIR}/lib
EIGEN3_ROOT=/usr/include
BOOST_ROOT=/usr/local/

export DEP_INCLUDE_DIR=${DEP_INCLUDE_DIR}
export DEP_INCLUDE_LIB=${DEP_INCLUDE_LIB}

function cleanup(){
	echo -e "\e[1m\e[31mCleaning up demo build directory ${BUILD_DIR}.\e[0m"
	rm -rf ${BUILD_DIR}
}

function setup_eigen(){
	echo -e "\e[1m\e[33mDownloading and setting up EIGEN.\e[0m"
	wget https://bitbucket.org/eigen/eigen/get/f3a22f35b044.zip -O ${DEPENDENCIES_DIR}/eigen_repo.zip --quiet
	unzip -qq ${DEPENDENCIES_DIR}/eigen_repo.zip -d ${DEPENDENCIES_DIR}
	mv ${DEPENDENCIES_DIR}/eigen-eigen-f3a22f35b044/Eigen ${DEP_INCLUDE_DIR}/Eigen
	rm -rf ${DEPENDENCIES_DIR}/eigen-eigen-f3a22f35b044
	rm -rf ${DEPENDENCIES_DIR}/eigen_repo.zip
	export EIGEN3_ROOT=${DEP_INCLUDE_DIR}
}

function setup_boost(){
	echo -e "\e[1m\e[33mDownloading and setting up BOOST.\e[0m"
	wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz -O ${DEPENDENCIES_DIR}/boost.tar.gz --quiet
	tar xzf ${DEPENDENCIES_DIR}/boost.tar.gz -C ${DEPENDENCIES_DIR}
	rm -rf ${DEPENDENCIES_DIR}/boost.tar.gz
	cd ${DEPENDENCIES_DIR}/boost_1_64_0/
	./bootstrap.sh --prefix=${DEPENDENCIES_DIR} -with-libraries=filesystem,system,iostreams,program_options
	./b2
	./b2 install &> out.log
	rm -rf ${DEPENDENCIES_DIR}/boost_1_64_0
	export BOOST_ROOT=${DEPENDENCIES_DIR}
	cd ${DEMO_ROOT}
}

function setup_plplot(){
	# Download 
	echo -e "\e[1m\e[33mInstalling gnuplot.\e[0m"
	sudo apt-get install gnuplot
	echo -e "\e[1m\e[33mDownloading and setting up PLPLOT.\e[0m"
	wget https://github.com/dstahlke/gnuplot-iostream/archive/master.zip -O ${DEPENDENCIES_DIR}/gnuplot++.zip --quiet
	unzip -o -qq ${DEPENDENCIES_DIR}/gnuplot++.zip -d ${DEPENDENCIES_DIR}
	rm -rf ${DEPENDENCIES_DIR}/gnuplot++.zip
	cp ${DEPENDENCIES_DIR}/gnuplot-iostream-master/gnuplot-iostream.h ${DEP_INCLUDE_DIR}
	rm -rf ${DEPENDENCIES_DIR}/gnuplot-iostream-master/
}

function setup_CImg(){
	# Download 
	echo -e "\e[1m\e[33mDownloading and setting up CImg.\e[0m"
	wget http://cimg.eu/files/CImg_latest.zip -O ${DEPENDENCIES_DIR}/CImg_latest.zip --quiet
	unzip -o -qq ${DEPENDENCIES_DIR}/CImg_latest.zip -d ${DEPENDENCIES_DIR}
	rm -rf ${DEPENDENCIES_DIR}/CImg_latest.zip
	cp ${DEPENDENCIES_DIR}/CImg-*/CImg.h ${DEP_INCLUDE_DIR}
	rm -rf ${DEPENDENCIES_DIR}/CImg-*/
}

function print_help(){
	local EXIT_CODE=$1

    echo "Usage: $0 [-h] [--clean] [--eigen | --boost | --all ] [--eigen_path] [--boost_path]"
    echo "Where:"
    echo "  --clean              : clean build directory (${BUILD_DIR})."
    echo "  --eigen              : download and setup eigen."
    echo "  --boost              : download and setup boost."
    echo "  --all                : download and setup both boost and eigen."
    echo "  --eigen_path         : root directory where to find eigen (where "
    echo "                         your Eigen/ folder is stored)."
    echo "  --boost_path         : root directory where to find boost (where you"
    echo "                         installed headers and libraries using the "
    echo "                         install scripts provided by the boost library)."
    echo "  -h                   : print this help."

    exit ${EXIT_CODE}
}

mkdir -p ${BUILD_DIR}


if [ $# -gt 0 ]
then
	# Check for cleanup flag
	if [ "$1" = "-h" ]
	then
		print_help;
	fi
fi

if [ $# -gt 0 ]
then
	# Check for cleanup flag
	if [ "$1" = "--clean" ]
	then
		cleanup;
		shift;
	fi
fi

mkdir -p ${DEPENDENCIES_DIR}
mkdir -p ${DEP_INCLUDE_DIR}

# Check which dependencies to install
while test $# -gt 0; do
	case $1 in
	    --eigen)
			setup_eigen;
	        shift;
	        ;;
	    --boost)
			setup_boost;
	        shift;
	        ;;
	    --all)
			setup_eigen;
			setup_boost;
	        shift;
	        ;;
        *)
            break
            ;;
	esac
done

# Check if path needs to be set
while test $# -gt 0; do
	case $1 in
	    --eigen_path)
			export EIGEN3_ROOT=$2
	        shift 2;
	        ;;
	    --boost_path)
			export BOOST_ROOT=$2
	        shift 2;
	        ;;
	    --all)
			setup_eigen;
			setup_boost;
	        shift;
	        ;;
        *)
            break
            ;;
	esac
done

# At this point preliminary setup is done
# Now we will just assume qt custom plot is not present and download it
setup_plplot
# same for CImg
setup_CImg

# Let's get some shit done
cd ${BUILD_DIR}
# export CXX="g++-4.9"
cmake -DCMAKE_BUILD_TYPE=Release ${DEMO_ROOT}
