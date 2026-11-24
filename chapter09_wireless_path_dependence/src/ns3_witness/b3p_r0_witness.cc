#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/propagation-module.h"
#include "bridge_shadowing_path_loss.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ns3;

namespace
{
constexpr double kSigma = 4.0;
constexpr double kSigma2 = 16.0;
constexpr double kDCor = 37.0;
constexpr double kTxPowerDbm = 30.0;
constexpr double kNoisePowerDbm = -87.0;
constexpr double kBandwidthHz = 1.0e8;
constexpr uint64_t kMasterSeed = 2026082801ULL;
constexpr int kWorldCount = 2000;

struct Mesh
{
    std::vector<Vector> point;
    std::vector<double> arclength;
    std::vector<double> incoming;
};

uint64_t
Mix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::mt19937_64
Engine(uint64_t world, uint64_t tag)
{
    return std::mt19937_64(Mix64(kMasterSeed ^ Mix64(world) ^ Mix64(tag)));
}

Mesh
MakeMesh(const std::vector<Vector>& route)
{
    Mesh mesh;
    mesh.point.push_back(route.front());
    mesh.arclength.push_back(0.0);
    mesh.incoming.push_back(0.0);
    double accumulated = 0.0;
    for (std::size_t segment = 0; segment + 1 < route.size(); ++segment)
    {
        const Vector p = route[segment];
        const Vector q = route[segment + 1];
        const Vector delta = q - p;
        const double length = delta.GetLength();
        double done = 0.0;
        while (done + 1.0 < length - 1e-11)
        {
            done += 1.0;
            mesh.point.push_back(p + delta * (done / length));
            mesh.arclength.push_back(accumulated + done);
            mesh.incoming.push_back(1.0);
        }
        const bool duplicate = (q - mesh.point.back()).GetLength() <= 1e-10;
        if (!duplicate)
        {
            mesh.point.push_back(q);
            mesh.arclength.push_back(accumulated + length);
            mesh.incoming.push_back(length - done);
        }
        accumulated += length;
    }
    return mesh;
}

std::vector<std::vector<double>>
Covariance(const Mesh& mesh)
{
    const std::size_t n = mesh.point.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            c[i][j] = kSigma2 * std::exp(-std::abs(mesh.arclength[i] - mesh.arclength[j]) / kDCor);
        }
    }
    return c;
}

std::vector<std::vector<double>>
Cholesky(const std::vector<std::vector<double>>& a)
{
    const std::size_t n = a.size();
    std::vector<std::vector<double>> l(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            double value = a[i][j];
            for (std::size_t k = 0; k < j; ++k)
            {
                value -= l[i][k] * l[j][k];
            }
            if (i == j)
            {
                if (value <= 0.0)
                {
                    throw std::runtime_error("non-positive Gaussian bridge covariance");
                }
                l[i][j] = std::sqrt(value);
            }
            else
            {
                l[i][j] = value / l[j][j];
            }
        }
    }
    return l;
}

struct Bridge
{
    std::vector<std::vector<double>> conditional;
    std::vector<std::vector<double>> cholesky;
    std::vector<double> coefficientS0;
    std::vector<double> coefficientST;
    double reconstructionError{0.0};
};

Bridge
MakeBridge(const std::vector<std::vector<double>>& c)
{
    const std::size_t n = c.size();
    const std::size_t m = n - 2;
    const double r = c.front().back() / kSigma2;
    const double inverseScale = 1.0 / (kSigma2 * (1.0 - r * r));
    Bridge bridge;
    bridge.conditional.assign(m, std::vector<double>(m));
    bridge.coefficientS0.resize(m);
    bridge.coefficientST.resize(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        const std::size_t ii = i + 1;
        bridge.coefficientS0[i] = inverseScale * (c[ii][0] - r * c[ii][n - 1]);
        bridge.coefficientST[i] = inverseScale * (c[ii][n - 1] - r * c[ii][0]);
        for (std::size_t j = 0; j < m; ++j)
        {
            const std::size_t jj = j + 1;
            const double reconstructed = c[ii][0] * bridge.coefficientS0[j] +
                                         c[ii][n - 1] * bridge.coefficientST[j];
            bridge.conditional[i][j] = c[ii][jj] - reconstructed;
            bridge.reconstructionError = std::max(bridge.reconstructionError,
                                                  std::abs(bridge.conditional[i][j] + reconstructed - c[ii][jj]));
        }
    }
    bridge.cholesky = Cholesky(bridge.conditional);
    return bridge;
}

std::vector<double>
SampleBridge(const Bridge& bridge, double s0, double st, const std::vector<double>& z)
{
    const std::size_t n = z.size() + 2;
    std::vector<double> state(n);
    state.front() = s0;
    state.back() = st;
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        double value = bridge.coefficientS0[i] * s0 + bridge.coefficientST[i] * st;
        for (std::size_t j = 0; j <= i; ++j)
        {
            value += bridge.cholesky[i][j] * z[j];
        }
        state[i + 1] = value;
    }
    return state;
}

std::vector<double>
Normals(uint64_t world, uint64_t tag, std::size_t count)
{
    auto engine = Engine(world, tag);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> values(count);
    for (double& value : values)
    {
        value = normal(engine);
    }
    return values;
}

double
RateBps(double rxPowerDbm)
{
    const double snrLinear = std::pow(10.0, (rxPowerDbm - kNoisePowerDbm) / 10.0);
    return kBandwidthHz * std::log2(1.0 + snrLinear);
}

struct Outcome
{
    std::vector<double> state;
    std::vector<double> rxPowerDbm;
    std::vector<double> rateBps;
    double endpointRxPowerDbm{0.0};
    double endpointRateBps{0.0};
    double endpointDeliveredBits{0.0};
    double routeDeliveredBits{0.0};
    double routeAverageRateBps{0.0};
    double routeOutageFraction{0.0};
};

Outcome
RunBranch(Ptr<ThreeGppUmaPropagationLossModel> model,
          Ptr<MobilityModel> bs,
          Ptr<ConstantPositionMobilityModel> ut,
          const Mesh& mesh,
          const std::vector<double>& state)
{
    b3p::BridgeShadowingPathLoss loss(model, state);
    Outcome outcome;
    outcome.state = state;
    double routeTime = 0.0;
    double outageTime = 0.0;
    for (std::size_t i = 0; i < mesh.point.size(); ++i)
    {
        ut->SetPosition(mesh.point[i]);
        const double rx = loss.CalcRxPowerDbm(kTxPowerDbm, bs, ut);
        const double rate = RateBps(rx);
        outcome.rxPowerDbm.push_back(rx);
        outcome.rateBps.push_back(rate);
        if (i > 0)
        {
            const double duration = mesh.incoming[i];
            outcome.routeDeliveredBits += duration * rate;
            routeTime += duration;
            if (rx < kNoisePowerDbm)
            {
                outageTime += duration;
            }
        }
        if (i + 1 == mesh.point.size())
        {
            outcome.endpointRxPowerDbm = rx;
            outcome.endpointRateBps = rate;
            outcome.endpointDeliveredBits = rate; // fixed 1-s endpoint window
        }
    }
    if (loss.GetConsumedMeshPoints() != mesh.point.size())
    {
        throw std::runtime_error("bridge wrapper did not consume exactly the explicit mesh");
    }
    outcome.routeAverageRateBps = outcome.routeDeliveredBits / routeTime;
    outcome.routeOutageFraction = outageTime / routeTime;
    return outcome;
}

struct MomentAccumulator
{
    std::size_t count{0};
    std::vector<double> sum;
    std::vector<std::vector<double>> outer;
    explicit MomentAccumulator(std::size_t n) : sum(n), outer(n, std::vector<double>(n)) {}
    void Add(const std::vector<double>& x)
    {
        ++count;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            sum[i] += x[i];
            for (std::size_t j = 0; j < x.size(); ++j)
            {
                outer[i][j] += x[i] * x[j];
            }
        }
    }
    std::vector<double> Mean() const
    {
        std::vector<double> answer(sum.size());
        for (std::size_t i = 0; i < sum.size(); ++i) answer[i] = sum[i] / count;
        return answer;
    }
    std::vector<std::vector<double>> SampleCovariance() const
    {
        const auto mean = Mean();
        std::vector<std::vector<double>> answer(sum.size(), std::vector<double>(sum.size()));
        for (std::size_t i = 0; i < sum.size(); ++i)
            for (std::size_t j = 0; j < sum.size(); ++j)
                answer[i][j] = (outer[i][j] - count * mean[i] * mean[j]) / (count - 1);
        return answer;
    }
};

void
WriteHeader(std::ofstream& out, const std::string& header)
{
    out << header << '\n';
    out << std::setprecision(17);
}

} // namespace

int
main(int argc, char** argv)
{
    if (argc != 2 || std::string(argv[1]).rfind("--output=", 0) != 0)
    {
        std::cerr << "usage: b3p_r0_witness --output=ABSOLUTE_OUTPUT_DIRECTORY\n";
        return 2;
    }
    const std::filesystem::path output = std::string(argv[1]).substr(std::string("--output=").size());
    std::filesystem::create_directories(output);

    const Vector bsPosition(0.0, 0.0, 25.0);
    const Vector x0(90.0, 0.0, 1.5);
    const Vector xa(100.0, 0.0, 1.5);
    const Vector waypoint(90.0, 30.1413273412, 1.5);
    const Vector xb(99.98, 1.9998999975, 1.5);
    const Mesh meshA = MakeMesh({x0, xa});
    const Mesh meshB = MakeMesh({x0, waypoint, xb});
    const auto covarianceA = Covariance(meshA);
    const auto covarianceB = Covariance(meshB);
    const Bridge bridgeA = MakeBridge(covarianceA);
    const Bridge bridgeB = MakeBridge(covarianceB);
    const double rA = std::exp(-10.0 / kDCor);
    const double rB = std::exp(-60.0 / kDCor);
    const double qA = std::sqrt(1.0 - rA * rA);
    const double qB = std::sqrt(1.0 - rB * rB);

    auto bsNode = CreateObject<Node>();
    auto utNode = CreateObject<Node>();
    auto bsMobility = CreateObject<ConstantPositionMobilityModel>();
    auto utMobility = CreateObject<ConstantPositionMobilityModel>();
    bsNode->AggregateObject(bsMobility);
    utNode->AggregateObject(utMobility);
    bsMobility->SetPosition(bsPosition);
    const auto condition = CreateObject<AlwaysLosChannelConditionModel>();
    const auto model = CreateObject<ThreeGppUmaPropagationLossModel>();
    model->SetAttribute("Frequency", DoubleValue(28e9));
    model->SetAttribute("ShadowingEnabled", BooleanValue(false));
    model->SetChannelConditionModel(condition);

    std::ofstream manifest(output / "registered_world_manifest.csv");
    std::ofstream endpoints(output / "endpoint_results.csv");
    std::ofstream operational(output / "operational_results.csv");
    std::ofstream trajectory(output / "trajectory_results.csv");
    WriteHeader(manifest, "world_id,master_seed,arms,shared_s0_substream,native_shadowing_enabled");
    WriteHeader(endpoints, "world_id,arm,kappa,s0_db,shadow_a_endpoint_db,shadow_b_endpoint_db,endpoint_difference_db,canonical_y_a,canonical_y_b,canonical_disagreement");
    WriteHeader(operational, "world_id,arm,branch,terminal_rx_power_dbm,terminal_rate_bps,terminal_delivered_bits,route_delivered_bits,route_average_rate_bps,route_outage_fraction");
    WriteHeader(trajectory, "world_id,arm,branch,mesh_index,x_m,y_m,z_m,cumulative_arclength_m,incoming_duration_s,shadowing_db,received_power_dbm,rate_bps");

    struct ArmAccumulator { std::string name; double kappa; MomentAccumulator a; MomentAccumulator b; };
    std::vector<ArmAccumulator> accumulators;
    for (const auto& definition : std::vector<std::pair<std::string, double>>{{"KAPPA_PLUS", 1.0}, {"KAPPA_ZERO", 0.0}, {"KAPPA_MINUS", -1.0}})
    {
        accumulators.push_back({definition.first, definition.second, MomentAccumulator(meshA.point.size()), MomentAccumulator(meshB.point.size())});
    }

    for (uint64_t world = 1; world <= kWorldCount; ++world)
    {
        const auto primitives = Normals(world, 1, 3);
        const double s0 = kSigma * primitives[0];
        const double ea = primitives[1];
        const double z = primitives[2];
        const auto bridgeNormalsA = Normals(world, 2, meshA.point.size() - 2);
        const auto bridgeNormalsB = Normals(world, 3, meshB.point.size() - 2);
        manifest << world << ',' << kMasterSeed << ",KAPPA_PLUS|KAPPA_ZERO|KAPPA_MINUS,"
                 << Mix64(kMasterSeed ^ Mix64(world) ^ Mix64(1)) << ",false\n";
        for (auto& arm : accumulators)
        {
            const double eb = arm.kappa == 1.0 ? ea : (arm.kappa == -1.0 ? -ea : z);
            const double stA = rA * s0 + kSigma * qA * ea;
            const double stB = rB * s0 + kSigma * qB * eb;
            const auto stateA = SampleBridge(bridgeA, s0, stA, bridgeNormalsA);
            const auto stateB = SampleBridge(bridgeB, s0, stB, bridgeNormalsB);
            const Outcome outcomeA = RunBranch(model, bsMobility, utMobility, meshA, stateA);
            const Outcome outcomeB = RunBranch(model, bsMobility, utMobility, meshB, stateB);
            arm.a.Add(stateA);
            arm.b.Add(stateB);
            const int yA = stA <= 0.0 ? 1 : 0;
            const int yB = stB <= 0.0 ? 1 : 0;
            endpoints << world << ',' << arm.name << ',' << arm.kappa << ',' << s0 << ',' << stA << ',' << stB << ','
                      << stA - stB << ',' << yA << ',' << yB << ',' << (yA != yB ? 1 : 0) << '\n';
            const auto writeOutcome = [&](const char* branch, const Outcome& result) {
                operational << world << ',' << arm.name << ',' << branch << ',' << result.endpointRxPowerDbm << ','
                            << result.endpointRateBps << ',' << result.endpointDeliveredBits << ','
                            << result.routeDeliveredBits << ',' << result.routeAverageRateBps << ','
                            << result.routeOutageFraction << '\n';
            };
            const auto writeTrajectory = [&](const char* branch, const Mesh& mesh, const Outcome& result) {
                for (std::size_t i = 0; i < mesh.point.size(); ++i)
                {
                    trajectory << world << ',' << arm.name << ',' << branch << ',' << i << ','
                               << mesh.point[i].x << ',' << mesh.point[i].y << ',' << mesh.point[i].z << ','
                               << mesh.arclength[i] << ',' << mesh.incoming[i] << ',' << result.state[i] << ','
                               << result.rxPowerDbm[i] << ',' << result.rateBps[i] << '\n';
                }
            };
            writeOutcome("A", outcomeA);
            writeOutcome("B", outcomeB);
            writeTrajectory("A", meshA, outcomeA);
            writeTrajectory("B", meshB, outcomeB);
        }
    }

    std::ofstream mechanism(output / "mechanism_validation.csv");
    WriteHeader(mechanism, "arm,branch,n,mesh_points,path_length_m,max_abs_state_mean,max_abs_covariance_error,max_relative_covariance_error,endpoint_mean,endpoint_variance,cov_s0_endpoint,target_cov_s0_endpoint,first_1m_adjacent_cov,target_first_1m_adjacent_cov,conditional_bridge_reconstruction_error,native_shadowing_enabled");
    const auto writeMechanism = [&](const ArmAccumulator& arm, const char* branch, const MomentAccumulator& moments,
                                    const Mesh& mesh, const std::vector<std::vector<double>>& target, const Bridge& bridge) {
        const auto mean = moments.Mean();
        const auto empirical = moments.SampleCovariance();
        double maxMean = 0.0, maxAbsolute = 0.0, maxRelative = 0.0;
        for (std::size_t i = 0; i < mean.size(); ++i)
        {
            maxMean = std::max(maxMean, std::abs(mean[i]));
            for (std::size_t j = 0; j < mean.size(); ++j)
            {
                const double error = std::abs(empirical[i][j] - target[i][j]);
                maxAbsolute = std::max(maxAbsolute, error);
                maxRelative = std::max(maxRelative, error / std::max(std::abs(target[i][j]), 1e-12));
            }
        }
        const std::size_t endpoint = mean.size() - 1;
        mechanism << arm.name << ',' << branch << ',' << moments.count << ',' << mesh.point.size() << ',' << mesh.arclength.back() << ','
                  << maxMean << ',' << maxAbsolute << ',' << maxRelative << ',' << mean[endpoint] << ','
                  << empirical[endpoint][endpoint] << ',' << empirical[0][endpoint] << ',' << target[0][endpoint] << ','
                  << empirical[0][1] << ',' << target[0][1] << ',' << bridge.reconstructionError << ",false\n";
    };
    for (const auto& arm : accumulators)
    {
        writeMechanism(arm, "A", arm.a, meshA, covarianceA, bridgeA);
        writeMechanism(arm, "B", arm.b, meshB, covarianceB, bridgeB);
    }
    std::cout << "registered worlds completed; operational outcomes not interpreted by executor\n";
    return 0;
}
