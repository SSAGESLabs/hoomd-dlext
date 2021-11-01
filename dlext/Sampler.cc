#include "Sampler.h"
#include "hoomd/HOOMDMath.h"
#include "dlpack.h"

using namespace std;
namespace py = pybind11;

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
template <>
constexpr int64_t stride1<int>() { return 1; }

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

void Sampler::update(unsigned int timestep)
{

  // Accessing the handles here holds them valid until the block of this function.
  // This keeps them valid for the python function call
  auto location = m_exec_conf->isCUDAEnabled() ? access_location::device : access_location::host;

  const ArrayHandle<Scalar4> pos(m_pdata->getPositions(), location, access_mode::read);
  auto pos_tensor = wrap(reinterpret_cast<Scalar*>(pos.data), 4 );
  ArrayHandle<Scalar4> vel(m_pdata->getVelocities(), location, access_mode::read);
  auto vel_tensor = wrap(reinterpret_cast<Scalar*>(vel.data), 4);
  ArrayHandle<unsigned int> rtags(m_pdata->getRTags(), location, access_mode::read);
  auto rtag_tensor = wrap(reinterpret_cast<unsigned int*>(rtags.data), 1);
  ArrayHandle<int3> img(m_pdata->getImages(), location, access_mode::read);
  auto img_tensor = wrap(reinterpret_cast<int*>(img.data), 3);

  ArrayHandle<Scalar4> net_forces(m_pdata->getNetForce(), location, access_mode::readwrite);
  auto force_tensor = wrap(reinterpret_cast<Scalar *>(net_forces.data), 4);

  m_python_update(pos_tensor, vel_tensor, rtag_tensor, img_tensor, force_tensor,
                  m_pdata->getGlobalBox());
}

template <typename T>
DLTensor Sampler::wrap(T* const ptr,
                       const int64_t size2,
                       const uint64_t offset,
                       uint64_t stride1_offset) {
  assert((size2 >= 1)); // assert is a macro so the extra parentheses are requiered here

  const unsigned int particle_number = this->m_pdata->getN();
  const bool on_device = this->m_exec_conf->isCUDAEnabled();
  const int gpu_id = m_exec_conf->isCUDAEnabled() ? m_exec_conf->getGPUIds()[0] : m_exec_conf->getRank();

  vector<int64_t> shape;
  shape.push_back(particle_number);
  if (size2 > 1)
    shape.push_back(size2);

  vector<int64_t> strides;
  strides.push_back(stride1<T>() + stride1_offset);
  if (size2 > 1)
    strides.push_back(1);

  DLTensor tensor;
  tensor.data = reinterpret_cast<void*>(ptr);
  tensor.ctx = DLContext{on_device ? kDLGPU : kDLCPU, gpu_id};

  tensor.ndim = shape.size();
  tensor.dtype = dtype<T>();
  tensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
  tensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
  tensor.byte_offset = offset;

  return tensor;
}


void export_Sampler(py::module& m)
{
  py::class_<Sampler, std::shared_ptr<Sampler> >(m, "CPPSampler", py::base<HalfStepHook>())
    .def(py::init<std::shared_ptr<SystemDefinition>, py::function>())
    ;
}
