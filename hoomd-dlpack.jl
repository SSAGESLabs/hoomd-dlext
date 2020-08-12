### A Pluto.jl notebook ###
# v0.11.4

using Markdown
using InteractiveUtils

# ╔═╡ 7516d9f4-c17a-11ea-1f2e-a5e6cd5fcec8
#=
"""
inline unsigned int getRTag(unsigned int tag) const
            {
            assert(tag < m_rtag.size());
            ArrayHandle< unsigned int> h_rtag(m_rtag,access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
#ifdef ENABLE_MPI
            assert(m_decomposition || idx < getN());
#endif
            assert(idx < getN() + getNGhosts() || idx == NOT_LOCAL);
            return idx;
            }
"""
=#

# ╔═╡ e962cf94-baf3-11ea-2c12-cf52180a8452
let

#function Base.convert(::Type{cxxt"$T"}, po::PyObject) where {T}
#    ptr = pcpp"PyObject"(Ptr{Cvoid}(PyPtr(po)))
#    return icxx"pybind11::handle($ptr).cast<$T>();"
#end

end

# ╔═╡ 7e8a2346-d675-11ea-1cf7-55043ece8767


# ╔═╡ 73d58072-d5cf-11ea-126e-871bc236d094
using Distributed

# ╔═╡ 1312be52-b7ee-11ea-0a1e-d585a4358cf5
using PyCall

# ╔═╡ 71a62c44-b7f1-11ea-2389-3d63eb374d69
using Conda

# ╔═╡ b48cc19a-b7f0-11ea-0ebe-cfa95dc5b573
@everywhere using Cxx

# ╔═╡ c3e29cc8-b7f0-11ea-157e-fb0a2dc0f5f7
begin

addHeaderDir(
    "/usr/lib/x86_64-linux-gnu/openmpi/include/";
    kind = C_System
)
addHeaderDir(
    joinpath(Conda.PREFIX, "include", "python3.7m");
    kind = C_System
)
addHeaderDir(
    joinpath(Conda.LIBDIR, "python3.7", "site-packages", "hoomd", "include");
    kind = C_System
)
addHeaderDir(
    joinpath(Conda.LIBDIR, "python3.7", "site-packages", "hoomd", "include");
    kind = C_User
)
addHeaderDir(
    joinpath(homedir(), "Documents", "Projects", "dlpack", "include");
    kind = C_User
)

end

# ╔═╡ 40276078-b7f3-11ea-3beb-fb618952faa2
hoomd = pyimport("hoomd");

# ╔═╡ 3bc45266-b7ee-11ea-2838-7ffc1066f897
md = pyimport("hoomd.md");

# ╔═╡ ccaee846-b7f4-11ea-3a64-ed8399a8ef9c
np = pyimport("numpy");

# ╔═╡ d3a83a24-b7f4-11ea-1e8d-f3d4c45a9004
begin
    kT = 0.596161
    Δt = 0.02045
end;

# ╔═╡ 5c3ba312-b7f5-11ea-279d-17a5e7e030cc
hoomd.context.initialize("--mode=cpu");

# ╔═╡ ca04adb6-b7f6-11ea-1bfe-cb7facc87c87
snapshot = hoomd.data.make_snapshot(
    N = 14,
    box = hoomd.data.boxdim(Lx = 41, Ly = 41, Lz = 41),
    particle_types = ["C", "H"],
    bond_types = ["CC", "CH"],
    angle_types = ["CCC", "CCH", "HCH"],
    dihedral_types = ["CCCC", "HCCC", "HCCH"],
    pair_types = ["CCCC", "HCCC", "HCCH"],
);

# ╔═╡ 44232068-b80c-11ea-1607-c5f6cd36c53a
begin

ids = PyArray(snapshot.particles."typeid")

ids[2:4] .= 1
ids[6:7] .= 1
ids[9:10] .= 1
ids[12:14] .= 1

end;

# ╔═╡ 7d5ff8ba-b80c-11ea-11c0-511138531794
positions = [
    -2.990196  0.097881  0.000091;
    -2.634894 -0.911406  0.001002;
    -2.632173  0.601251 -0.873601;
    -4.060195  0.099327 -0.000736;
    -2.476854  0.823942  1.257436;
    -2.832157  1.833228  1.256526;
    -2.834877  0.320572  2.131128;
    -0.936856  0.821861  1.258628;
    -0.578833  1.325231  0.384935;
    -0.581553 -0.187426  1.259538;
    -0.423514  1.547922  2.515972;
    -0.781537  1.044552  3.389664;
     0.646485  1.546476  2.516800;
    -0.778816  2.557208  2.515062;
];

# ╔═╡ b049bbca-b7f4-11ea-2c2c-7db511e51ec3
reference_box_low_coords = [ -22.206855 -19.677099 -19.241968 ];

# ╔═╡ 0afa234e-b80d-11ea-31d7-f3f3f78cb99b
box_low_coords = [ -snapshot.box.Lx/2 -snapshot.box.Ly/2 -snapshot.box.Lz/2 ];

# ╔═╡ 195f7828-b80d-11ea-310e-13df174a5f69
positions .+= box_low_coords .- reference_box_low_coords;

# ╔═╡ 2035fd3c-b80d-11ea-181e-bdb5eeff1d32
PyArray(snapshot.particles."position") .= positions;

# ╔═╡ 2e28a1ce-b80d-11ea-3302-6101814cb1ac
begin

mC = 12.00
mH = 1.008
PyArray(snapshot.particles."mass") .= [
    mC, mH, mH, mH,
    mC, mH, mH,
    mC, mH, mH,
    mC, mH, mH, mH
];

end;

# ╔═╡ 26f587ca-b81a-11ea-2c8f-2bf56d5f80df
begin

reference_charges = [
    -0.180000, 0.060000, 0.060000, 0.060000,
    -0.120000, 0.060000, 0.060000,
    -0.120000, 0.060000, 0.060000,
    -0.180000, 0.060000, 0.060000, 0.060000
]
charge_conversion = 18.22262

PyArray(snapshot.particles."charge") .= charge_conversion .* reference_charges

end;

# ╔═╡ 3ca1a8ba-b81a-11ea-0c30-ab1ff884531a
begin

snapshot.bonds.resize(13)

bonds_ids = PyArray(snapshot.bonds."typeid")

bonds_ids[1:3] .= 1
bonds_ids[5:6] .= 1
bonds_ids[8:9] .= 1
bonds_ids[11:13] .= 1

end;

# ╔═╡ 4cbf062a-b81a-11ea-0229-532364a68323
PyArray(snapshot.bonds."group") .= [
     0   2;
     0   1;
     0   3;
     0   4;
     4   5;
     4   6;
     4   7;
     7   8;
     7   9;
     7  10;
    10  11;
    10  12;
    10  13;
];

# ╔═╡ 66e38ff8-b81a-11ea-19e5-3f5113993571
begin

snapshot.angles.resize(24)

angles_ids = PyArray(snapshot.angles."typeid")

angles_ids[1:2] .= 2
angles_ids[3] = 1
angles_ids[4] = 2
angles_ids[5:8] .= 1
angles_ids[9] = 0
angles_ids[10] = 2
angles_ids[11:14] .= 1
angles_ids[15] = 0
angles_ids[16] = 2
angles_ids[17:21] .= 1
angles_ids[22:24] .= 2

end;

# ╔═╡ 7601e4da-b81a-11ea-005c-4dfd0d30dd34
PyArray(snapshot.angles."group") .= [
     1   0   2;
     2   0   3;
     2   0   4;
     1   0   3;
     1   0   4;
     3   0   4;
     0   4   5;
     0   4   6;
     0   4   7;
     5   4   6;
     5   4   7;
     6   4   7;
     4   7   8;
     4   7   9;
     4   7  10;
     8   7   9;
     8   7  10;
     9   7  10;
     7  10  11;
     7  10  12;
     7  10  13;
    11  10  12;
    11  10  13;
    12  10  13;
];

# ╔═╡ 7e912a0a-b81a-11ea-0d9b-4b4c88af3b33
begin

snapshot.dihedrals.resize(27)

dihedrals_ids = PyArray(snapshot.dihedrals."typeid")

dihedrals_ids[1:2] .= 2
dihedrals_ids[3] = 1
dihedrals_ids[4:5] .= 2
dihedrals_ids[6] = 1
dihedrals_ids[7:8] .= 2
dihedrals_ids[9:11] .= 1
dihedrals_ids[12] = 0
dihedrals_ids[13:14] .= 2
dihedrals_ids[15] = 1
dihedrals_ids[16:17] .= 2
dihedrals_ids[18:21] .= 1
dihedrals_ids[22:27] .= 2

end;

# ╔═╡ 944b3aae-b81a-11ea-2655-e986b1f80c5a
PyArray(snapshot.dihedrals."group") .= [
    2  0   4   5;
    2  0   4   6;
    2  0   4   7;
    1  0   4   5;
    1  0   4   6;
    1  0   4   7;
    3  0   4   5;
    3  0   4   6;
    3  0   4   7;
    0  4   7   8;
    0  4   7   9;
    0  4   7  10;
    5  4   7   8;
    5  4   7   9;
    5  4   7  10;
    6  4   7   8;
    6  4   7   9;
    6  4   7  10;
    4  7  10  11;
    4  7  10  12;
    4  7  10  13;
    8  7  10  11;
    8  7  10  12;
    8  7  10  13;
    9  7  10  11;
    9  7  10  12;
    9  7  10  13;
];

# ╔═╡ 9b84b61a-b81a-11ea-1a2e-19fae8fc8d71
begin

snapshot.pairs.resize(27)

pairs_ids = PyArray(snapshot.pairs."typeid")

pairs_ids[1:1] .= 0
pairs_ids[2:11] .= 1
pairs_ids[12:27] .= 2

PyArray(snapshot.pairs."group") .= [
    # CCCC
     0  10;
    # HCCC
     0   8;
     0   9;
     5  10;
     6  10;
     1   7;
     2   7;
     3   7;
    11   4;
    12   4;
    13   4;
    # HCCH
     1   5;
     1   6;
     2   5;
     2   6;
     3   5;
     3   6;
     5   8;
     6   8;
     5   9;
     6   9;
     8  11;
     8  12;
     8  13;
     9  11;
     9  12;
     9  13;
]

end;

# ╔═╡ c75a39c0-b81a-11ea-1ce3-3b6c0d5b020e
system = hoomd.init.read_snapshot(snapshot);

# ╔═╡ 2d3bc4e4-c177-11ea-308d-d14a4851f73a
pdata = system.particles.pdata

# ╔═╡ 6705f758-b89e-11ea-3eb3-8f7b6bf933dd
cxx"""

//#define ENABLE_CUDA
#define ENABLE_MPI
#define ENABLE_TBB

#if !defined(DEFAULT_VISIBILITY)
#if defined(WIN32) || defined(_WIN32)
#define DEFAULT_VISIBILITY __declspec(dllexport)
#else
#define DEFAULT_VISIBILITY __attribute__((visibility("default")))
#endif
#endif

#define INVOKE(object, member_ptr) ((object).*(member_ptr))

template <typename T>
constexpr void maybe_unused(T&&) { }

"""

# ╔═╡ f75bcb4c-b81d-11ea-2e14-03d63e294885
cxx"""

#include <memory>
#include <type_traits>
#include <vector>

#include "dlpack/dlpack.h"

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/SystemDefinition.h"

"""

# ╔═╡ 12f8f5aa-be31-11ea-08c0-2b0b92639304
cxx"""

using DLManagedTensorPtr = DLManagedTensor*;

using ParticleDataPtr = std::shared_ptr<ParticleData>;
using SystemDefinitionPtr = std::shared_ptr<SystemDefinition>;
using ExecutionConfigurationPtr = std::shared_ptr<const ExecutionConfiguration>;

using AccessLocation = access_location::Enum;
constexpr auto kOnHost = access_location::host;
#ifdef ENABLE_CUDA
constexpr auto kOnDevice = access_location::device;
#endif

using AccessMode = access_mode::Enum;
constexpr auto kRead = access_mode::read;
constexpr auto kReadWrite = access_mode::readwrite;
constexpr auto kOverwrite = access_mode::overwrite;

constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

"""

# ╔═╡ 7958e50c-b827-11ea-3281-7b9d16a6fedf
cxx"""

class DEFAULT_VISIBILITY SystemView {
public:
    SystemView(SystemDefinitionPtr sysdef);
    ParticleDataPtr particle_data() const;
    ExecutionConfigurationPtr exec_config() const;
    bool is_gpu_enabled() const;
    unsigned int particle_number() const;
    int get_device_id(bool gpu_flag) const;
private:
    SystemDefinitionPtr sysdef;
    ParticleDataPtr pdata;
    ExecutionConfigurationPtr exec_conf;
};

"""

# ╔═╡ 5cedc368-d804-11ea-347c-a351efdc4d38
cxx"""

SystemView::SystemView(SystemDefinitionPtr sysdef)
    : sysdef { sysdef }
    , pdata { sysdef->getParticleData() }
{
    exec_conf = pdata->getExecConf();
}

ParticleDataPtr SystemView::particle_data() const { return pdata; }
ExecutionConfigurationPtr SystemView::exec_config() const { return exec_conf; }
bool SystemView::is_gpu_enabled() const { return exec_conf->isCUDAEnabled(); }
unsigned int SystemView::particle_number() const { return pdata->getN(); }

int SystemView::get_device_id(bool gpu_flag) const {
    maybe_unused(gpu_flag); // prevent compiler warnings when ENABLE_CUDA is not defined
#ifdef ENABLE_CUDA
    if (gpu_flag)
        return exec_conf->getGPUIds()[0];
#endif
    return exec_conf->getRank();
}

"""

# ╔═╡ 3a44c2a6-d088-11ea-17d3-37feacc734ca
cxx"""

template <template <typename> class A, typename T, typename Object>
using PropertyGetter = const A<T>& (Object::*)() const;

template <typename T>
using ArrayHandlePtr = std::unique_ptr<ArrayHandle<T>>;

"""

# ╔═╡ 06d49ad2-bbba-11ea-26dc-73b026f0389a
cxx"""

template <typename T>
struct DLDataBridge {
    ArrayHandlePtr<T> handle;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;

    DLDataBridge(ArrayHandlePtr<T>& handle)
        : handle(std::move(handle))
    { }
};

template <typename T>
using DLDataBridgePtr = std::unique_ptr<DLDataBridge<T>>;

template <typename T>
void DLDataBridgeDeleter(DLManagedTensorPtr tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge<T>*>(tensor->manager_ctx);
}

"""

# ╔═╡ b120feb8-d802-11ea-1be9-5be8158f0568
cxx"""

template <typename T>
void* opaque(T* data) { return static_cast<void*>(data); }

DLContext context(const SystemView& sysview, bool gpu_flag)
{
    return DLContext { gpu_flag ? kDLGPU : kDLCPU, sysview.get_device_id(gpu_flag) };
}

constexpr DLDataType dtype(const DLDataBridgePtr<Scalar4>&)
{
    return DLDataType {kDLFloat, kBits, 1};
}
constexpr DLDataType dtype(const DLDataBridgePtr<Scalar3>&)
{
    return DLDataType {kDLFloat, kBits, 1};
}
constexpr DLDataType dtype(const DLDataBridgePtr<Scalar>&)
{
    return DLDataType {kDLFloat, kBits, 1};
}
constexpr DLDataType dtype(const DLDataBridgePtr<int3>&)
{
    return DLDataType {kDLInt, 32, 1};
}
constexpr DLDataType dtype(const DLDataBridgePtr<unsigned int>&)
{
    return DLDataType {kDLUInt, 32, 1};
}

constexpr int64_t stride1(const DLDataBridgePtr<Scalar4>&) { return 4; }
constexpr int64_t stride1(const DLDataBridgePtr<Scalar3>&) { return 3; }
constexpr int64_t stride1(const DLDataBridgePtr<Scalar>&) { return 1; }
constexpr int64_t stride1(const DLDataBridgePtr<int3>&) { return 3; }
constexpr int64_t stride1(const DLDataBridgePtr<unsigned int>&) { return 1; }

"""

# ╔═╡ d0cb9c82-d802-11ea-1845-b17abf41f5e5
cxx"""

template <template <typename> class A, typename T, typename O>
DLManagedTensorPtr wrap(
    const SystemView& sysview, PropertyGetter<A, T, O> getter,
    AccessLocation requested_location, AccessMode mode,
    int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0
)
{
    assert((size2 >= 1));

    auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = ArrayHandlePtr<T>(
        new ArrayHandle<T>(INVOKE(*(sysview.particle_data()), getter)(), location, mode)
    );
    auto bridge = DLDataBridgePtr<T>(new DLDataBridge<T>(handle));

#ifdef ENABLE_CUDA
    auto gpu_flag = (location == kOnDevice);
#else
    auto gpu_flag = false;
#endif

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = DLDataBridgeDeleter<T>;

    auto& dltensor = bridge->tensor.dl_tensor;
    dltensor.data = opaque(bridge->handle->data);
    dltensor.ctx = context(sysview, gpu_flag);
    dltensor.dtype = dtype(bridge);

    auto& shape = bridge->shape;
    shape.push_back(sysview.particle_number());
    if (size2 > 1) shape.push_back(size2);

    auto& strides = bridge->strides;
    strides.push_back(stride1(bridge) + stride1_offset);
    if (size2 > 1) strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset;

    return &(bridge.release()->tensor);
}

"""

# ╔═╡ 551f5afc-c177-11ea-15e8-47faa6954a5b
let
    ptr = pcpp"PyObject"(Ptr{Cvoid}(PyPtr(pdata.getPosition(0))))
    icxx"""
        auto h = pybind11::handle {$ptr};
        h.cast<Scalar3>();
    """
end

# ╔═╡ 3c6df83a-c178-11ea-2e7c-a72eb69916ef
let
    ptr = pcpp"PyObject"(Ptr{Cvoid}(PyPtr(pdata)))
    icxx"""
        auto h = pybind11::handle {$ptr};
        auto pdata = h.cast<ParticleDataPtr>();
        pdata->getPosition(0);
    """
end

# ╔═╡ 81364030-c125-11ea-0776-a1aedbb2405e
sysdef = let
    ptr = pcpp"PyObject"(Ptr{Cvoid}(PyPtr(system.sysdef)))
    icxx"""
        auto h = pybind11::handle($ptr);
        h.cast<SystemDefinitionPtr>();
    """
end

#sysdef = convert(cxxt"SystemDefinitionPtr", system.sysdef)

# ╔═╡ 184bd918-bba8-11ea-130b-696c874fabac
sv = icxx"auto sv = SystemView($sysdef); sv;"

# ╔═╡ 98ef35b8-c127-11ea-3e5d-7b33619f8eaa
icxx"""

const auto pdata = $sysdef->getParticleData();

{
    auto foo = ArrayHandle<Scalar4>(
        pdata->getPositions(), kOnHost, kReadWrite
    );

    //std::cout << foo.data[0].x << std::endl;
}

pdata->getN();

"""

# ╔═╡ 9bb0a2f8-d803-11ea-1d9c-410b6abc4239
tensor_ptr = icxx"wrap($sv, &ParticleData::getPositions, kOnHost, kReadWrite, 3);"

# ╔═╡ 172ab8fa-c211-11ea-03cb-e97701a245cd
icxx"$tensor_ptr->dl_tensor;"

# ╔═╡ 7a67a59c-d803-11ea-2c0e-4be9e1d9e6ee
let
    N = Int(icxx"$tensor_ptr->dl_tensor.ndim;")
    ptr = icxx"$tensor_ptr->dl_tensor.shape;"
    unsafe_load(Ptr{NTuple{N,Int64}}(ptr))
end

# ╔═╡ 77985708-d803-11ea-2ad9-dbffeb2c5a38
let
    N = Int(icxx"$tensor_ptr->dl_tensor.ndim;")
    ptr = icxx"$tensor_ptr->dl_tensor.strides;"
    unsafe_load(Ptr{NTuple{N,Int64}}(ptr))
end

# ╔═╡ 3d3789b4-d676-11ea-1c0f-b7ce9b350ba3
icxx"""

auto wt = wrap(
    $sv, &ParticleData::getPositions, kOnHost, kReadWrite, 1, 3
);
wt->dl_tensor;

"""

# ╔═╡ eaddaa8c-dc50-11ea-0403-c1c7f3c0194d
cxx"""

DLManagedTensorPtr positions(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getPositions, location, mode, 3);
}
DLManagedTensorPtr types(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getPositions, location, mode, 1, 3);
}
DLManagedTensorPtr velocities(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getVelocities, location, mode, 3);
}
DLManagedTensorPtr masses(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getVelocities, location, mode, 1, 3);
}
DLManagedTensorPtr orientations(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getOrientationArray, location, mode, 4);
}
DLManagedTensorPtr angular_momenta(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getAngularMomentumArray, location, mode, 4);
}
DLManagedTensorPtr moments_of_intertia(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getMomentsOfInertiaArray, location, mode, 3);
}
DLManagedTensorPtr charges(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getCharges, location, mode, 1);
}
DLManagedTensorPtr diameters(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getDiameters, location, mode, 1);
}
DLManagedTensorPtr images(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getImages, location, mode, 3);
}
DLManagedTensorPtr tags(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getTags, location, mode, 1);
}
DLManagedTensorPtr net_forces(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetForce, location, mode, 4);
}
DLManagedTensorPtr net_torques(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetTorqueArray, location, mode, 4);
}
DLManagedTensorPtr net_virial(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetVirial, location, mode, 6, 0, 5);
}

"""

# ╔═╡ 7dd89158-dc51-11ea-29cf-5f31bd492b58
icxx"images($sv, kOnHost, kReadWrite)->dl_tensor;"

# ╔═╡ Cell order:
# ╠═73d58072-d5cf-11ea-126e-871bc236d094
# ╠═1312be52-b7ee-11ea-0a1e-d585a4358cf5
# ╠═71a62c44-b7f1-11ea-2389-3d63eb374d69
# ╠═b48cc19a-b7f0-11ea-0ebe-cfa95dc5b573
# ╠═c3e29cc8-b7f0-11ea-157e-fb0a2dc0f5f7
# ╠═40276078-b7f3-11ea-3beb-fb618952faa2
# ╠═3bc45266-b7ee-11ea-2838-7ffc1066f897
# ╠═ccaee846-b7f4-11ea-3a64-ed8399a8ef9c
# ╠═d3a83a24-b7f4-11ea-1e8d-f3d4c45a9004
# ╠═5c3ba312-b7f5-11ea-279d-17a5e7e030cc
# ╠═ca04adb6-b7f6-11ea-1bfe-cb7facc87c87
# ╠═44232068-b80c-11ea-1607-c5f6cd36c53a
# ╠═7d5ff8ba-b80c-11ea-11c0-511138531794
# ╠═b049bbca-b7f4-11ea-2c2c-7db511e51ec3
# ╠═0afa234e-b80d-11ea-31d7-f3f3f78cb99b
# ╠═195f7828-b80d-11ea-310e-13df174a5f69
# ╠═2035fd3c-b80d-11ea-181e-bdb5eeff1d32
# ╠═2e28a1ce-b80d-11ea-3302-6101814cb1ac
# ╠═26f587ca-b81a-11ea-2c8f-2bf56d5f80df
# ╠═3ca1a8ba-b81a-11ea-0c30-ab1ff884531a
# ╠═4cbf062a-b81a-11ea-0229-532364a68323
# ╠═66e38ff8-b81a-11ea-19e5-3f5113993571
# ╠═7601e4da-b81a-11ea-005c-4dfd0d30dd34
# ╠═7e912a0a-b81a-11ea-0d9b-4b4c88af3b33
# ╠═944b3aae-b81a-11ea-2655-e986b1f80c5a
# ╠═9b84b61a-b81a-11ea-1a2e-19fae8fc8d71
# ╠═c75a39c0-b81a-11ea-1ce3-3b6c0d5b020e
# ╠═2d3bc4e4-c177-11ea-308d-d14a4851f73a
# ╠═6705f758-b89e-11ea-3eb3-8f7b6bf933dd
# ╠═f75bcb4c-b81d-11ea-2e14-03d63e294885
# ╠═12f8f5aa-be31-11ea-08c0-2b0b92639304
# ╠═7958e50c-b827-11ea-3281-7b9d16a6fedf
# ╠═5cedc368-d804-11ea-347c-a351efdc4d38
# ╠═3a44c2a6-d088-11ea-17d3-37feacc734ca
# ╠═06d49ad2-bbba-11ea-26dc-73b026f0389a
# ╠═b120feb8-d802-11ea-1be9-5be8158f0568
# ╠═d0cb9c82-d802-11ea-1845-b17abf41f5e5
# ╠═551f5afc-c177-11ea-15e8-47faa6954a5b
# ╠═7516d9f4-c17a-11ea-1f2e-a5e6cd5fcec8
# ╠═3c6df83a-c178-11ea-2e7c-a72eb69916ef
# ╠═e962cf94-baf3-11ea-2c12-cf52180a8452
# ╠═81364030-c125-11ea-0776-a1aedbb2405e
# ╠═184bd918-bba8-11ea-130b-696c874fabac
# ╠═98ef35b8-c127-11ea-3e5d-7b33619f8eaa
# ╠═9bb0a2f8-d803-11ea-1d9c-410b6abc4239
# ╠═172ab8fa-c211-11ea-03cb-e97701a245cd
# ╠═7a67a59c-d803-11ea-2c0e-4be9e1d9e6ee
# ╠═77985708-d803-11ea-2ad9-dbffeb2c5a38
# ╠═3d3789b4-d676-11ea-1c0f-b7ce9b350ba3
# ╠═eaddaa8c-dc50-11ea-0403-c1c7f3c0194d
# ╠═7dd89158-dc51-11ea-29cf-5f31bd492b58
# ╠═7e8a2346-d675-11ea-1cf7-55043ece8767
