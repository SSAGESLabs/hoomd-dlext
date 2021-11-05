#include "Sampler.h"
#include "hoomd/HOOMDMath.h"
#include "dlpack.h"
#include <stdexcept>

using namespace std;
namespace py = pybind11;

const char* const kDLTensorCapsuleName = "dltensor";
constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

template <typename>
constexpr DLDataType dtype();
template <>
constexpr DLDataType dtype<Scalar4>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<Scalar3>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<Scalar>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<int3>() { return DLDataType {kDLInt, 32, 1}; }
template <>
constexpr DLDataType dtype<unsigned int>() { return DLDataType {kDLUInt, 32, 1}; }
template <>
constexpr DLDataType dtype<int>() { return DLDataType {kDLInt, 32, 1}; }

template <typename>
constexpr int64_t stride1();
template <>
constexpr int64_t stride1<Scalar4>() { return 4; }
template <>
constexpr int64_t stride1<Scalar3>() { return 3; }
template <>
constexpr int64_t stride1<Scalar>() { return 1; }
template <>
constexpr int64_t stride1<int3>() { return 3; }
template <>
constexpr int64_t stride1<unsigned int>() { return 1; }

template <typename T>
inline void* opaque(T* data) { return static_cast<void*>(data); }

inline py::capsule encapsulate(DLManagedTensor* dl_managed_tensor)
{
  return py::capsule(dl_managed_tensor, kDLTensorCapsuleName);
}

Sampler::Sampler(shared_ptr<SystemDefinition> sysdef,
                 py::function python_update)
  :
  HalfStepHook(),
  m_python_update(python_update)
{
  this->setSystemDefinition(sysdef);
}

void Sampler::setSystemDefinition(shared_ptr<SystemDefinition> sysdef)
{
  m_sysdef = sysdef;
  m_pdata = sysdef->getParticleData();
  m_exec_conf = m_pdata->getExecConf();
}

void Sampler::run_on_data(py::function py_exec, const access_location::Enum location, const access_mode::Enum mode)
{
  if(location == access_location::device and not m_exec_conf->isCUDAEnabled())
    throw runtime_error("Invalid request for device memory in non-cuda run.");

  const bool on_device = location == access_location::device;

  const ArrayHandle<Scalar4> pos(m_pdata->getPositions(), location, mode);
  auto pos_bridge = wrap<Scalar4, Scalar>(pos.data, on_device, 4 );
  auto pos_capsule = encapsulate(&pos_bridge.tensor);

  const ArrayHandle<Scalar4> vel(m_pdata->getVelocities(), location, mode);
  auto vel_bridge = wrap<Scalar4, Scalar>(vel.data, on_device, 4 );
  auto vel_capsule = encapsulate(&vel_bridge.tensor);

  const ArrayHandle<unsigned int> rtags(m_pdata->getRTags(), location, mode);
  auto rtags_bridge = wrap<unsigned int, unsigned int>(rtags.data, on_device, 1);
  auto rtags_capsule = encapsulate(&rtags_bridge.tensor);

  const ArrayHandle<int3> img(m_pdata->getImages(), location, mode);
  auto img_bridge = wrap<int3, int>(img.data, on_device, 3);
  auto img_capsule = encapsulate(&img_bridge.tensor);

  ArrayHandle<Scalar4> force(m_pdata->getNetForce(), location, access_mode::readwrite);
  auto force_bridge = wrap<Scalar4, Scalar>(force.data, on_device, 4 );
  auto force_capsule = encapsulate(&force_bridge.tensor);

  py_exec(pos_capsule, vel_capsule, rtags_capsule, img_capsule, force_capsule);
}

void Sampler::update(unsigned int timestep)
{

  // Accessing the handles here holds them valid until the block of this function.
  // This keeps them valid for the python function call
  auto location = m_exec_conf->isCUDAEnabled() ? access_location::device : access_location::host;

  // const ArrayHandle<Scalar4> pos(m_pdata->getPositions(), location, access_mode::read);
  // auto pos_tensor = wrap<Scalar4, Scalar>(pos.data, 4 );
  // ArrayHandle<Scalar4> vel(m_pdata->getVelocities(), location, access_mode::read);
  // auto vel_tensor = wrap<Scalar4, Scalar>(vel.data, 4);
  // ArrayHandle<unsigned int> rtags(m_pdata->getRTags(), location, access_mode::read);
  // auto rtag_tensor = wrap<unsigned int, unsigned int>(rtags.data, 1);
  // ArrayHandle<int3> img(m_pdata->getImages(), location, access_mode::read);
  // auto img_tensor = wrap<int3, int>(img.data, 3);

  // ArrayHandle<Scalar4> net_forces(m_pdata->getNetForce(), location, access_mode::readwrite);
  // auto force_tensor = wrap<Scalar4, Scalar>(net_forces.data, 4);

  // m_python_update(pos_tensor, vel_tensor, rtag_tensor, img_tensor, force_tensor,
  //                 m_pdata->getGlobalBox());
  this->run_on_data(m_python_update, location, access_mode::read);
}

template <typename TV, typename TS>
DLDataBridge Sampler::wrap(TV* ptr,
                           const bool on_device,
                           const int64_t size2,
                           const uint64_t offset,
                           uint64_t stride1_offset) {
  assert((size2 >= 1)); // assert is a macro so the extra parentheses are requiered here

  const unsigned int particle_number = this->m_pdata->getN();
  const int gpu_id = on_device ? m_exec_conf->getGPUIds()[0] : m_exec_conf->getRank();

  DLDataBridge bridge;
  bridge.tensor.manager_ctx = NULL;
  bridge.tensor.deleter = NULL;

  bridge.tensor.dl_tensor.data = opaque(ptr);
  bridge.tensor.dl_tensor.ctx = DLContext{on_device ? kDLGPU : kDLCPU, gpu_id};
  bridge.tensor.dl_tensor.dtype = dtype<TS>();

  bridge.shape.push_back(particle_number);
  if (size2 > 1)
    bridge.shape.push_back(size2);

  bridge.strides.push_back(stride1<TV>() + stride1_offset);
  if (size2 > 1)
    bridge.strides.push_back(1);

  bridge.tensor.dl_tensor.ndim = bridge.shape.size();
  bridge.tensor.dl_tensor.dtype = dtype<TS>();
  bridge.tensor.dl_tensor.shape = reinterpret_cast<std::int64_t*>(bridge.shape.data());
  bridge.tensor.dl_tensor.strides = reinterpret_cast<std::int64_t*>(bridge.strides.data());
  bridge.tensor.dl_tensor.byte_offset = offset;

  return bridge;
}


void export_Sampler(py::module& m)
{
  py::class_<Sampler, std::shared_ptr<Sampler> >(m, "DLextSampler", py::base<HalfStepHook>())
    .def(py::init<std::shared_ptr<SystemDefinition>, py::function>())
    .def("run_on_data", &Sampler::run_on_data)
    ;
}
