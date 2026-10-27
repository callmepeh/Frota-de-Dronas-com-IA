import java.util.*;
import java.util.function.Consumer;

class Drone {
    final int id;
    final boolean rented;
    final Queue<Package> queue;
    final SimScheduler scheduler;
    final Map<String, Double> missionSteps = new LinkedHashMap<>();
    DroneState state;
    /** Compatibility shim for SimScheduler_BACKUP; production code uses state. */
    @Deprecated boolean isAvalaible;
    double lastAdExp;
    private Consumer<Double> onDelivery;

    Drone(int id, boolean rented, DroneState initialState, Queue<Package> queue, SimScheduler scheduler) {
        this.id = id; this.rented = rented; this.state = initialState; this.isAvalaible = initialState == DroneState.AVAILABLE; this.queue = queue; this.scheduler = scheduler;
        missionSteps.put("POUSO_DECOLA1", 30_000.0); missionSteps.put("VOO1", 1_200_000.0);
        missionSteps.put("POUSO_DECOLA2", 30_000.0); missionSteps.put("POUSO_DECOLA3", 30_000.0);
        missionSteps.put("VOO2", 1_200_000.0); missionSteps.put("POUSO_DECOLA4", 30_000.0);
    }
    void setOnDelivery(Consumer<Double> callback) { onDelivery = callback; }
    void tryPickup() {
        if (state != DroneState.AVAILABLE) return;
        Package p = queue.poll(); if (p == null) return;
        lastAdExp = p.adExp; state = DroneState.ACTIVE; p.startServiceTime = scheduler.now();
        final double queueTime = p.startServiceTime - p.arrivalTime;
        final Iterator<Map.Entry<String, Double>> stages = missionSteps.entrySet().iterator(); final double[] flightTime = {0};
        Runnable nextStage = new Runnable() { public void run() {
            if (stages.hasNext()) { Map.Entry<String, Double> stage = stages.next(); double duration = Utils.generateExponential(stage.getValue()); flightTime[0] += duration; scheduler.schedule("Drone" + id, stage.getKey(), duration, duration, this); }
            else { double missionTime = queueTime + flightTime[0]; Utils.addMissionTime(missionTime); Utils.tp++; if (onDelivery != null) onDelivery.accept(missionTime); scheduler.missionCompleted(Drone.this); }
        }};
        nextStage.run();
    }
}
