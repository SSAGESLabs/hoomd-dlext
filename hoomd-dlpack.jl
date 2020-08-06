### A Pluto.jl notebook ###
# v0.11.2

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

# ╔═╡ baf4053a-d73e-11ea-2758-19426f007406


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

using AccessLocation = access_location::Enum;
#ifdef ENABLE_CUDA
constexpr auto kOnDevice = access_location::device;
#endif
constexpr auto kOnHost = access_location::host;

using AccessMode = access_mode::Enum;
constexpr auto kReadWrite = access_mode::readwrite;

"""

# ╔═╡ c6227b30-d73e-11ea-085f-25891cf667f5
cxx"""

template <typename T>
constexpr void maybe_unused(T&&) { }

"""

# ╔═╡ 7958e50c-b827-11ea-3281-7b9d16a6fedf
cxx"""

using ParticleDataPtr = std::shared_ptr<ParticleData>;
using SystemDefinitionPtr = std::shared_ptr<SystemDefinition>;
using ExecutionConfigurationPtr = std::shared_ptr<const ExecutionConfiguration>;

class DEFAULT_VISIBILITY SystemView {
public:
    SystemView(SystemDefinitionPtr sysdef);
    ParticleDataPtr particles_data() const ;
    ExecutionConfigurationPtr exec_config() const;
    bool is_gpu_enabled() const ;
    uint8_t precision_bits() const;
    unsigned int particles_number() const;
private:
    SystemDefinitionPtr sysdef;
    ParticleDataPtr pdata;
    ExecutionConfigurationPtr exec_conf;
    uint8_t bits;
};

SystemView::SystemView(SystemDefinitionPtr sysdef)
    : sysdef { sysdef }
    , pdata { sysdef->getParticleData() }
{
    exec_conf = pdata->getExecConf();
    bits = std::is_same<Scalar, float>::value ? 32 : 64;
}

ParticleDataPtr SystemView::particles_data() const { return pdata; }
ExecutionConfigurationPtr SystemView::exec_config() const { return exec_conf; }
bool SystemView::is_gpu_enabled() const { return exec_conf->isCUDAEnabled(); }
uint8_t SystemView::precision_bits() const { return bits; }
unsigned int SystemView::particles_number() const { return pdata->getN(); }

"""

# ╔═╡ 3a44c2a6-d088-11ea-17d3-37feacc734ca
cxx"""

template <typename T>
using PropertyGetter = const GlobalArray<T>& (ParticleData::*)() const;

"""

# ╔═╡ 06d49ad2-bbba-11ea-26dc-73b026f0389a
cxx"""

using DLManagedTensorPtr = DLManagedTensor*;

template <typename T>
using ArrayHandlePtr = std::unique_ptr<ArrayHandle<T>>;

template <typename T>
struct DLDataBridge {
    ArrayHandlePtr<T> handle;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;

    DLDataBridge(ArrayHandlePtr<T>& handle) : handle { std::move(handle) } { }
};

template <typename T>
using DLDataBridgePtr = std::unique_ptr<DLDataBridge<T>>;

template <typename T>
void DLDataBridgeDeleter(DLManagedTensorPtr tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge<T>*>(tensor->manager_ctx);
}

template <typename T>
void* opaque(T* data) { return static_cast<void*>(data); }
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

# ╔═╡ f782f8be-bbbf-11ea-37a4-f9806ed27d55
cxx"""
void set(DLContext& ctx, DLDeviceType device_type, int device_id = 0)
{
    ctx.device_type = device_type;
    ctx.device_id = device_id;
}

void set(DLDataType& dtype, uint8_t code, uint8_t bits, uint16_t lanes)
{
    dtype.code = code;
    dtype.bits = bits;
    dtype.lanes = lanes;
}
"""

# ╔═╡ c054b476-c20c-11ea-3d14-ff8b11518dae
cxx"""

int64_t stride(const GlobalArray<Scalar4>&) { return 4; }
int64_t stride(const GlobalArray<Scalar3>&) { return 3; }
int64_t stride(const GlobalArray<Scalar>&) { return 1; }

"""

# ╔═╡ c77020d4-cb75-11ea-0f23-370c21efa1cd
cxx"""
int gpu_id(const SystemView& sysview) {
#ifdef ENABLE_CUDA
    return sysview.exec_config()->getGPUIds()[0];
#else
    return -1;
#endif
}
"""

# ╔═╡ f12fa1fc-cc41-11ea-0f65-8d9e1b01dc35
icxx"gpu_id($sv);"

# ╔═╡ ff2000e8-cd3d-11ea-0e0d-e37b3841ea42
cxx"""
int get_id(const SystemView& sysview, AccessLocation location) {
    return (location == kOnHost) ? sysview.exec_config()->getRank() : gpu_id(sysview);
}
"""

# ╔═╡ a0bec5b6-d5a8-11ea-24b2-eb532695c09e
cxx"""

constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

constexpr uint8_t bits(const DLDataBridgePtr<Scalar4>&) { return kBits; }
constexpr uint8_t bits(const DLDataBridgePtr<Scalar3>&) { return kBits; }
constexpr uint8_t bits(const DLDataBridgePtr<Scalar>&) { return kBits; }
constexpr uint8_t bits(const DLDataBridgePtr<int3>&) { return 32; }
constexpr uint8_t bits(const DLDataBridgePtr<unsigned int>&) { return 32; }

"""

# ╔═╡ 02074c74-bbde-11ea-336a-f9534a74bd76
function wrap(
	sysview,
    array::cxxt"const GlobalArray<$T>&", #"
	requested_location::cxxt"AccessLocation",
    mode::cxxt"AccessMode",
	dtype_code::cxxt"DLDataTypeCode",
	ndim, size2, offset
) where {T}
    icxx"""
		auto location = $sysview.is_gpu_enabled() ? $requested_location : kOnHost;
        auto handle = ArrayHandlePtr<$T>(new ArrayHandle<$T>($array, location, $mode));
        auto bridge = DLDataBridgePtr<$T>(new DLDataBridge<$T>(handle));

        bridge->tensor.manager_ctx = bridge.get();
        bridge->tensor.deleter = DLDataBridgeDeleter<$T>;

        auto& dltensor = bridge->tensor.dl_tensor;
        dltensor.data = opaque(bridge->handle->data);

        auto dev_type = (location == kOnHost) ? kDLCPU : kDLGPU;
		auto dev_id = (location == kOnHost) ?
			$sysview.exec_config()->getRank() : gpu_id($sysview);

        set(dltensor.ctx, dev_type, dev_id);

        set(dltensor.dtype, $dtype_code, bits(bridge), 1);

        auto& shape = bridge->shape;
        auto& strides = bridge->strides;

		shape.push_back($sysview.particles_number());
		if ($ndim == 2)
			shape.push_back($size2);
		
		strides.push_back(stride($array));
		if ($ndim == 2)
        	strides.push_back(1);

        dltensor.ndim = shape.size();
        dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
        dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
        dltensor.byte_offset = $offset;

        &(bridge.release()->tensor);
    """
end

# ╔═╡ a10544dc-be38-11ea-0e42-654b5ac0bf67
tensor_ptr = wrap(
    sv, icxx"$sv.particles_data()->getPositions();",
	icxx"kOnHost;", icxx"kReadWrite;",
	icxx"kDLFloat;", 2, 3, 0
)

# ╔═╡ 172ab8fa-c211-11ea-03cb-e97701a245cd
icxx"$tensor_ptr->dl_tensor;"

# ╔═╡ 9d5f6f24-c26b-11ea-09ed-036d30fe3af9
let
    N = Int(icxx"$tensor_ptr->dl_tensor.ndim;")
    ptr = icxx"$tensor_ptr->dl_tensor.shape;"
    unsafe_load(Ptr{NTuple{N,Int64}}(ptr))
end

# ╔═╡ f984ec80-bfc2-11ea-2da8-a1c9060d33c8
let
    N = Int(icxx"$tensor_ptr->dl_tensor.ndim;")
    ptr = icxx"$tensor_ptr->dl_tensor.strides;"
    unsafe_load(Ptr{NTuple{N,Int64}}(ptr))
end

# ╔═╡ 33ec28fc-d096-11ea-2956-db61fb8d059a
cxx"""

constexpr int64_t stride(const DLDataBridgePtr<Scalar4>&) { return 4; }
constexpr int64_t stride(const DLDataBridgePtr<Scalar3>&) { return 3; }
constexpr int64_t stride(const DLDataBridgePtr<Scalar>&) { return 1; }
constexpr int64_t stride(const DLDataBridgePtr<int3>&) { return 3; }
constexpr int64_t stride(const DLDataBridgePtr<unsigned int>&) { return 1; }

"""

# ╔═╡ 407dbe42-d5ed-11ea-3617-136b21c8dcfe
cxx"""

template <template <typename> class A, typename T>
using Getter = const A<T>& (ParticleData::*)() const;

"""

# ╔═╡ 87a30418-d66e-11ea-1321-d9721c2d86aa
cxx"""

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

"""

# ╔═╡ dfdfcf90-d690-11ea-0e15-e9045c7eb871
cxx"""

int get_id(const SystemView& sysview, bool on_host) {
#ifdef ENABLE_CUDA
	if !(on_host)
		return sysview.exec_config()->getGPUIds()[0];
#endif
    return sysview.exec_config()->getRank();
}

"""

# ╔═╡ 6c659e3a-d679-11ea-3576-6b71eccc332d
cxx"""
DLContext context(const SystemView& sysview, AccessLocation location) {
	auto on_host = (location == kOnHost);
    return DLContext {
		on_host ? kDLCPU : kDLGPU,
		get_id(sysview, on_host)
	};
}
"""

# ╔═╡ 97501a52-d094-11ea-0c5e-d7704e08a3f1
cxx"""

template <template <typename> class A, typename T>
DLManagedTensorPtr wrap(
	const SystemView& sysview, Getter<A, T> getter,
    AccessLocation requested_location, AccessMode mode,
	int64_t size2, uint64_t offset = 0
) {
	auto pdata = sysview.particles_data();
	auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = ArrayHandlePtr<T>(
		new ArrayHandle<T>(INVOKE(*pdata, getter)(), location, mode)
	);
    auto bridge = DLDataBridgePtr<T>(new DLDataBridge<T>(handle));

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = DLDataBridgeDeleter<T>;

	auto& dltensor = bridge->tensor.dl_tensor;
    dltensor.data = opaque(bridge->handle->data);
    dltensor.ctx = context(sysview, location);
    dltensor.dtype = dtype(bridge);

    auto& shape = bridge->shape;
    auto& strides = bridge->strides;

	shape.push_back(sysview.particles_number());
	if (size2 > 1)
		shape.push_back(size2);
		
	strides.push_back(stride(bridge));
	if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset;

    return &(bridge.release()->tensor);
}

"""

# ╔═╡ f65b515e-d5ed-11ea-1ee5-0bc27d2ee073
icxx"""

auto wt = wrap(
	$sv, &ParticleData::getPositions, kOnHost, kReadWrite, 3
);
wt->dl_tensor;

"""

# ╔═╡ 3d3789b4-d676-11ea-1c0f-b7ce9b350ba3
icxx"""

auto wt = wrap(
	$sv, &ParticleData::getPositions, kOnHost, kReadWrite, 1, 3
);
wt->dl_tensor;

"""

# ╔═╡ a7b7b0da-d67a-11ea-0101-d941e02580f1
let wt = icxx"wrap($sv, &ParticleData::getPositions, kOnHost, kReadWrite, 1, 3);"
	N = Int(icxx"$wt->dl_tensor.ndim;")
	shape_ptr = icxx"$wt->dl_tensor.shape;"
	strides_ptr = icxx"$wt->dl_tensor.strides;"
	(
		unsafe_load(Ptr{NTuple{N,Int64}}(shape_ptr)),
		unsafe_load(Ptr{NTuple{N,Int64}}(strides_ptr)),
	)
end

# ╔═╡ ab0dab18-d670-11ea-0237-f3a61249f038
icxx"""
/*
auto wt = wrap(
	$sv, &ParticleData::getImages, kOnHost, kReadWrite, 3
);
wt->dl_tensor;
*/
"""

# ╔═╡ 577c1064-d5e1-11ea-3bb2-b95f5bd84b68
icxx"""
/*
auto wt = wrap(
	$sv, &ParticleData::getBodies, kOnHost, kReadWrite, 1
);
wt->dl_tensor;
*/
"""

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
# ╠═c6227b30-d73e-11ea-085f-25891cf667f5
# ╠═7958e50c-b827-11ea-3281-7b9d16a6fedf
# ╠═3a44c2a6-d088-11ea-17d3-37feacc734ca
# ╠═06d49ad2-bbba-11ea-26dc-73b026f0389a
# ╠═551f5afc-c177-11ea-15e8-47faa6954a5b
# ╠═7516d9f4-c17a-11ea-1f2e-a5e6cd5fcec8
# ╠═3c6df83a-c178-11ea-2e7c-a72eb69916ef
# ╠═e962cf94-baf3-11ea-2c12-cf52180a8452
# ╠═81364030-c125-11ea-0776-a1aedbb2405e
# ╠═184bd918-bba8-11ea-130b-696c874fabac
# ╠═98ef35b8-c127-11ea-3e5d-7b33619f8eaa
# ╠═f782f8be-bbbf-11ea-37a4-f9806ed27d55
# ╠═baf4053a-d73e-11ea-2758-19426f007406
# ╠═c054b476-c20c-11ea-3d14-ff8b11518dae
# ╠═c77020d4-cb75-11ea-0f23-370c21efa1cd
# ╠═f12fa1fc-cc41-11ea-0f65-8d9e1b01dc35
# ╠═ff2000e8-cd3d-11ea-0e0d-e37b3841ea42
# ╠═a0bec5b6-d5a8-11ea-24b2-eb532695c09e
# ╠═02074c74-bbde-11ea-336a-f9534a74bd76
# ╠═a10544dc-be38-11ea-0e42-654b5ac0bf67
# ╠═172ab8fa-c211-11ea-03cb-e97701a245cd
# ╠═9d5f6f24-c26b-11ea-09ed-036d30fe3af9
# ╠═f984ec80-bfc2-11ea-2da8-a1c9060d33c8
# ╠═33ec28fc-d096-11ea-2956-db61fb8d059a
# ╠═407dbe42-d5ed-11ea-3617-136b21c8dcfe
# ╠═87a30418-d66e-11ea-1321-d9721c2d86aa
# ╠═dfdfcf90-d690-11ea-0e15-e9045c7eb871
# ╠═6c659e3a-d679-11ea-3576-6b71eccc332d
# ╠═97501a52-d094-11ea-0c5e-d7704e08a3f1
# ╠═f65b515e-d5ed-11ea-1ee5-0bc27d2ee073
# ╠═3d3789b4-d676-11ea-1c0f-b7ce9b350ba3
# ╠═a7b7b0da-d67a-11ea-0101-d941e02580f1
# ╠═ab0dab18-d670-11ea-0237-f3a61249f038
# ╠═577c1064-d5e1-11ea-3bb2-b95f5bd84b68
# ╠═7e8a2346-d675-11ea-1cf7-55043ece8767
