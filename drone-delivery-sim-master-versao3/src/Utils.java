import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadLocalRandom;

public class Utils {
    public static SimScheduler scheduler;
    public static CountDownLatch latch;

    // --- Métricas globais ---
    public static double totalMissionTime = 0;
    public static double lastMissionTime = 0;       // ✅ tempo real da última missão
    public static double meanMissionTime = 0;       // (mantido apenas para relatórios finais)
    public static int deliveredPackagesNumber = 0;
    public static int generatedPackagesNumber = 0;
    public static int allocatedInQueuePackagesNumber = 0;
    public static int dropedPackagesNumber = 0;
    public static int tp = 0;
    public static Queue<Package> packageQueue;
    public static int minDrones;
    public static int maxDrones;
    public static List<Drone> listaDeDrones = new ArrayList<>();
    public static int numDronesAtivos;

    private static final Random rand = new Random();

    // --- Atualiza métricas com valor real ---
    public static synchronized void addMissionTime(double missionTime) {
        totalMissionTime += missionTime;
        deliveredPackagesNumber++;
        lastMissionTime = missionTime; // ✅ grava o valor real mais recente
        meanMissionTime = totalMissionTime / deliveredPackagesNumber; // ainda calcula média para relatórios finais
    }

    public static int getDeliveredPackagesNumber() {
        return deliveredPackagesNumber; // 👈 retorna o valor atual
    }

    // ✅ retorna o valor real mais recente
    public static synchronized double getLastMissionTime() {
        return lastMissionTime;
    }

    public static synchronized double getMeanMissionTime() {
        return meanMissionTime;
    }

    public static synchronized void registerDeliveryTimes(double missionTime, double queueTime, double flightTime) {
        // Local para armazenar métricas adicionais se necessário
    }

    // --- Reset das variáveis ---
    public static void clearVariables() {
        totalMissionTime = 0;
        lastMissionTime = 0;
        meanMissionTime = 0;
        deliveredPackagesNumber = 0;
        generatedPackagesNumber = 0;
        allocatedInQueuePackagesNumber = 0;
        dropedPackagesNumber = 0;
        tp = 0;
        listaDeDrones.clear();
    }

    public static void setLatch(CountDownLatch l) {
        latch = l;
    }

    public static double generateExponential(double mean) {
        return -Math.log(1 - ThreadLocalRandom.current().nextDouble()) * mean;
    }

    // --- Exibe estatísticas finais ---
    public static void printResults() {
        System.out.printf(Locale.US,
                "\n>> Pacotes gerados: %d\n" +
                        ">> Alocados: %d\n" +
                        ">> Entregues: %d\n" +
                        ">> Dropados: %d\n" +
                        ">> MMT Final (média): %.4f ms\n" +
                        ">> Último tempo de missão: %.4f ms\n",
                generatedPackagesNumber,
                allocatedInQueuePackagesNumber,
                deliveredPackagesNumber,
                dropedPackagesNumber,
                meanMissionTime,
                lastMissionTime
        );
    }
}
