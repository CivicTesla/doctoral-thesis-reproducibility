#ifndef B3P_BRIDGE_SHADOWING_PATH_LOSS_H
#define B3P_BRIDGE_SHADOWING_PATH_LOSS_H

#include "ns3/mobility-model.h"
#include "ns3/boolean.h"
#include "ns3/three-gpp-propagation-loss-model.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace b3p
{

/**
 * Minimal, non-invasive shadowing injector for the R0 explicit mesh.
 *
 * The contained ns-3 UMa model is deliberately configured with its native
 * ShadowingEnabled attribute set to false.  It supplies deterministic LOS
 * path loss only; exactly one pre-generated Gaussian-bridge shadowing value
 * is then subtracted at each monotonically consumed mesh index.  The vector
 * cursor is the wrapper's only cache/state.  It is intentionally separate
 * from ns-3's native shadowing cache, which is inactive in this construction.
 */
class BridgeShadowingPathLoss
{
  public:
    BridgeShadowingPathLoss(ns3::Ptr<ns3::ThreeGppUmaPropagationLossModel> deterministicModel,
                            std::vector<double> shadowingDb)
        : m_model(deterministicModel),
          m_shadowingDb(std::move(shadowingDb))
    {
        if (!m_model)
        {
            throw std::invalid_argument("deterministic UMa model is required");
        }
        // Make native shadowing inactive by construction, rather than relying
        // on a caller to remember a separate configuration step.
        m_model->SetAttribute("ShadowingEnabled", ns3::BooleanValue(false));
    }

    double CalcRxPowerDbm(double txPowerDbm,
                          ns3::Ptr<ns3::MobilityModel> a,
                          ns3::Ptr<ns3::MobilityModel> b)
    {
        if (m_nextMeshIndex == m_shadowingDb.size())
        {
            throw std::out_of_range("no bridge shadowing value for mesh evaluation");
        }
        const double deterministicRx = m_model->CalcRxPower(txPowerDbm, a, b);
        return deterministicRx - m_shadowingDb.at(m_nextMeshIndex++);
    }

    std::size_t GetConsumedMeshPoints() const
    {
        return m_nextMeshIndex;
    }

  private:
    ns3::Ptr<ns3::ThreeGppUmaPropagationLossModel> m_model;
    std::vector<double> m_shadowingDb;
    std::size_t m_nextMeshIndex{0};
};

} // namespace b3p

#endif // B3P_BRIDGE_SHADOWING_PATH_LOSS_H
