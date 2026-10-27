
/**
 * Representa um evento na simulação, com tempo e ação a ser executada.
 */
class SimEvent implements Comparable<SimEvent> {
    String proxyName;
    String name;
    double value;
    double time;
    Runnable action;

    SimEvent(String proxyName, String name, double value, double time, Runnable action) {
        this.proxyName = proxyName;
        this.name = name;
        this.value = value;
        this.time = time;
        this.action = action;
    }

    @Override
    public int compareTo(SimEvent other) {
        return Double.compare(this.time, other.time);
    }
}
