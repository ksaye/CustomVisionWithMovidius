#-------------------------------------------------------------------------
# Copyright(C) 2019 Intel Corporation.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Ubuntu 18.04 has some challenges with libboost 1.58.x
FROM ubuntu:16.04

#Device         Target Device
#======         =============
#CPU_FP32	    Intel CPUs
#GPU_FP32	    Intel Integrated Graphics
#GPU_FP16	    Intel Integrated Graphics
#MYRIAD_FP16	Intel MovidiusTM USB sticks (aka Intel® Neural Compute Stick 2 (Intel® NCS 2) or the original Intel® Movidius™ NCS)
#VAD-M_FP16	    Intel Vision Accelerator Design based on MovidiusTM MyriadX VPUs
ARG DEVICE=VAD-M_FP16
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=master
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz

WORKDIR /code
ARG MY_ROOT=/code
ENV PATH /opt/miniconda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:$PATH

ENV pattern="COMPONENTS=DEFAULTS"
ENV replacement="COMPONENTS=intel-openvino-ie-sdk-ubuntu-xenial__x86_64;intel-openvino-ie-rt-cpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-gpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-vpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-hddl-ubuntu-xenial__x86_64;intel-openvino-model-optimizer__x86_64;intel-openvino-opencv-lib-ubuntu-xenial__x86_64"
ENV LD_LIBRARY_PATH=/opt/miniconda/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_2020.2.120
ENV InferenceEngine_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/share
ENV IE_PLUGINS_PATH=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64
ENV LD_LIBRARY_PATH=/opt/intel/opencl:${INTEL_OPENVINO_DIR}/inference_engine/external/gna/lib:${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/omp/lib:${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/lib:${IE_PLUGINS_PATH}:${LD_LIBRARY_PATH}
ENV OpenCV_DIR=${INTEL_OPENVINO_DIR}/opencv/share/OpenCV
ENV LD_LIBRARY_PATH=${INTEL_OPENVINO_DIR}/opencv/lib:${INTEL_OPENVINO_DIR}/opencv/share/OpenCV/3rdparty/lib:${LD_LIBRARY_PATH}
ENV HDDL_INSTALL_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/hddl
ENV LD_LIBRARY_PATH=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/hddl/lib:$LD_LIBRARY_PATH
ENV LANG en_US.UTF-8

RUN apt update && \
    apt -y install git sudo wget locales \
    zip x11-apps lsb-core cpio libboost-python-dev libpng-dev zlib1g-dev libnuma1 ocl-icd-libopencl1 clinfo libboost-filesystem1.58.0 libboost-thread1.58.0 protobuf-compiler libprotoc-dev libusb-1.0-0-dev autoconf automake libtool && \
    cd ${MY_ROOT} && \
    git clone --recursive -b $ONNXRUNTIME_BRANCH $ONNXRUNTIME_REPO onnxruntime && \
    cp onnxruntime/docs/Privacy.md /code/Privacy.md && \
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt && \
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt && \
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh && \
    cd onnxruntime/cmake/external/onnx && python3 setup.py install && \
    pip install azure-iothub-device-client azure-iothub-service-client azure-iot-provisioning-device-client && \
    cd ${MY_ROOT} && \
    curl -LOJ "${DOWNLOAD_LINK}" && \
    tar -xzf l_openvino_toolkit*.tgz && \
    rm -rf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd - && \
    rm -rf l_openvino_toolkit* && \
    cd /opt/intel/openvino/install_dependencies && ./_install_all_dependencies.sh && dpkg -i *.deb && \
    cd ${MY_ROOT} && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    cd ${MY_ROOT} && \
    cd onnxruntime && ./build.sh --config Release --update --build --parallel --use_openvino $DEVICE --build_wheel --use_full_protobuf && \
    pip install build/Linux/Release/dist/*-linux_x86_64.whl && rm -rf /code/onnxruntime /code/cmake-3.14.3-Linux-x86_64

#    sed -i "s/$pattern/$replacement/" silent.cfg && \

# from this point on, this is used to add and serve the CustomVision.ai onnx files
ARG CustomVisionModelLink=https://kevinsayazstorage.blob.core.windows.net/public/d64f43ac870a437497ce47721d95301d.ONNX.zip
ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y install tzdata libssl-dev libffi-dev libgtk2.0-dev python3-pip && \
    pip install --upgrade pip && \
    pip install opencv-python flask pytz pillow && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata && \
    cd ${MY_ROOT} && \
    curl -LOJ "${CustomVisionModelLink}" && \
    unzip *.ONNX.zip && \
    rm -rf *.ONNX.zip

COPY main.py python/

EXPOSE 87

CMD [ "python3", "-u", "python/main.py" ]
