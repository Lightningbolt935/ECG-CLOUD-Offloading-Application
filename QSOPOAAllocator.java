package org.fog.test.perfeval;

import java.util.*;

public class QSOPOAAllocator {

    private List<Integer> edgeNodeIds = new ArrayList<>();
    private int cloudNodeId = -1;

    // Current load on each edge node (fraction 0-1)
    private Map<Integer, Double> nodeLoad = new HashMap<>();

    private static final double SATURATION_THRESHOLD = 0.80;
    private static final int    QSO_POPULATION       = 20;
    private static final int    QSO_ITERATIONS       = 30;
    private static final int    POA_ITERATIONS       = 15;
    private static final double QSO_ALPHA            = 0.5;

    public void setEdgeNodes(List<Integer> ids) {
        this.edgeNodeIds = ids;
        for (int id : ids) nodeLoad.put(id, 0.0);
    }

    public void setCloudNodeId(int id) { this.cloudNodeId = id; }

    // Main allocation method — called for each incoming task
    public int allocate(long taskMI, double taskBW, int taskClass) {
        // Critical task -> cloud immediately
        if (taskClass == 3) return cloudNodeId;

        // Get available (non-saturated) edge nodes
        List<Integer> available = getAvailableNodes();

        // All saturated -> cloud overflow
        if (available.isEmpty()) return cloudNodeId;

        // QSO: global exploration
        int qsoBest = qsoExplore(taskMI, taskBW, available);

        // POA: local refinement
        int poaBest = poaRefine(qsoBest, taskMI, taskBW, available);

        // Update load
        updateLoad(poaBest, taskMI);

        return poaBest;
    }

    private int qsoExplore(long mi, double bw, List<Integer> available) {
        // Initialize population
        List<Integer> pop = new ArrayList<>();
        Random rnd = new Random();
        for (int i = 0; i < QSO_POPULATION; i++) {
            pop.add(available.get(rnd.nextInt(available.size())));
        }

        int    gBest    = pop.get(0);
        double gBestFit = fitness(gBest, mi, bw);

        for (int iter = 0; iter < QSO_ITERATIONS; iter++) {
            // Build probability weights
            double[] weights = new double[available.size()];
            double totalW = 0;
            for (int i = 0; i < available.size(); i++) {
                weights[i] = 1.0 / (fitness(available.get(i), mi, bw) + 1e-9);
                totalW += weights[i];
            }
            for (int i = 0; i < weights.length; i++) weights[i] /= totalW;

            // Update population
            List<Integer> newPop = new ArrayList<>();
            for (int nodeId : pop) {
                int newNode;
                if (rnd.nextDouble() < QSO_ALPHA) {
                    // Social learning: sample from weighted distribution
                    newNode = weightedSample(available, weights, rnd);
                } else {
                    newNode = available.get(rnd.nextInt(available.size()));
                }
                newPop.add(newNode);
                double f = fitness(newNode, mi, bw);
                if (f < gBestFit) { gBest = newNode; gBestFit = f; }
            }
            pop = newPop;
        }
        return gBest;
    }

    private int poaRefine(int startNode, long mi, double bw,
                          List<Integer> available) {
        int    best    = startNode;
        double bestFit = fitness(best, mi, bw);
        double step    = 0.3;

        for (int iter = 0; iter < POA_ITERATIONS; iter++) {
            boolean improved = false;
            for (int nid : available) {
                if (nid == best) continue;
                double f = fitness(nid, mi, bw);
                if (f < bestFit - step * 0.01) {
                    best    = nid;
                    bestFit = f;
                    improved = true;
                }
            }
            step *= 0.9;
            if (!improved) break;
        }
        return best;
    }

    private double fitness(int nodeId, long mi, double bw) {
        double load   = nodeLoad.getOrDefault(nodeId, 0.0);
        double effMI  = mi * 0.05; // 5% pre-screening
        double latency = (effMI / 1000.0) * 1000.0 + 1.0; // ms
        double energy  = effMI * 0.02;
        double loadPenalty = Math.max(0, (load - 0.5) / 0.5);
        return 0.5 * (latency / 270.0) + 0.3 * (energy / 5.0) + 0.2 * loadPenalty;
    }

    private List<Integer> getAvailableNodes() {
        List<Integer> available = new ArrayList<>();
        for (int id : edgeNodeIds) {
            if (nodeLoad.getOrDefault(id, 0.0) < SATURATION_THRESHOLD) {
                available.add(id);
            }
        }
        return available;
    }

    private void updateLoad(int nodeId, long mi) {
        double current = nodeLoad.getOrDefault(nodeId, 0.0);
        // Simple load model: MI processed / node capacity per window
        double capacity = 1000.0 * 500.0; // MIPS * batch_duration_sec
        nodeLoad.put(nodeId, Math.min(current + (mi * 0.05 / capacity), 1.0));
    }

    private int weightedSample(List<Integer> items, double[] weights, Random rnd) {
        double r = rnd.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < items.size(); i++) {
            cumulative += weights[i];
            if (r <= cumulative) return items.get(i);
        }
        return items.get(items.size() - 1);
    }
}