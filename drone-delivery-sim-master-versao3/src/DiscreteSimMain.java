import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Properties;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class DiscreteSimMain {

    static double ad; // valor base (arrival delay)
    static int maxQueueSize;
    static int numDrones;
    static int minimoNumDrones;
    static double simulationTime = 72_000_000; // 86_400_000 = 1 dia   108_000_000 ~= 1 dia e 5 horas

    static boolean dynamicAD = false;
    static boolean debugAD = false;
    static double adMin, adMax;
    static double upTime, lowTime, downTime, highTime;
    static SimScheduler.ExecutionMode executionMode = SimScheduler.ExecutionMode.DYNAMIC_DT;
    static double rechargeTime = 15_000;
    static double displacementTime = 15_000;
    static double dtTargetMmt = 2_400_000;
    static int dtIdleReserve = 2;
    static double dtDecisionInterval = 300_000;

    private static final double AWAIT_TIMEOUT_MULTIPLIER = 2.0;

    static DatabaseManager monitorDT = DatabaseManager.getInstance();

    public static void main(String[] args) throws IOException {
        Locale.setDefault(Locale.US);
        loadProperties("config.properties");

        System.out.println(">> Simulação iniciada com " + numDrones + " drones.");
        System.out.println(">> Tempo total (ms): " + simulationTime);
        System.out.println(">> Dynamic AD: " + dynamicAD + (dynamicAD ? String.format(" (min=%.2f max=%.2f)", adMin, adMax) : ""));

        // CSV para AD x MMT
        BufferedWriter writer = new BufferedWriter(new FileWriter("ad_mmt_log.csv", false));
        writer.write("Time(ms);CurrentAD;MeanMissionTime\n");

        exec(numDrones, writer);

        // encerra writer (caso não tenha sido fechado)
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(">> Resultados salvos em ad_mmt_log.csv");
        System.exit(0);
    }

    private static void loadProperties(String filename) throws IOException {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(filename)) {
            props.load(fis);
        }

        numDrones = Integer.parseInt(props.getProperty("numDrones").trim());
        minimoNumDrones = Integer.parseInt(props.getProperty("minimoNumDrones").trim());
        executionMode = SimScheduler.ExecutionMode.valueOf(props.getProperty("executionMode", "DYNAMIC_DT").trim().toUpperCase(Locale.ROOT));
        rechargeTime = Double.parseDouble(props.getProperty("RECARGA", "15000").trim());
        displacementTime = Double.parseDouble(props.getProperty("DESLOCAMENTO", "15000").trim());
        dtTargetMmt = Double.parseDouble(props.getProperty("DT_TARGET_MMT", "2400000").trim());
        dtIdleReserve = Integer.parseInt(props.getProperty("DT_IDLE_RESERVE", "2").trim());
        dtDecisionInterval = Double.parseDouble(props.getProperty("DT_DECISION_INTERVAL", "300000").trim());
        Utils.numDronesAtivos = executionMode == SimScheduler.ExecutionMode.STATIC ? numDrones : minimoNumDrones;
        ad = Double.parseDouble(props.getProperty("ad").trim());
        maxQueueSize = Integer.parseInt(props.getProperty("maxQueueSize").trim());
        //simulationTime = Double.parseDouble(props.getProperty("simulationTime"));

        Utils.minDrones = minimoNumDrones;
        Utils.maxDrones = numDrones;

        if (props.containsKey("adMin")) {
            dynamicAD = true;
            adMin = Double.parseDouble(props.getProperty("adMin"));
            adMax = Double.parseDouble(props.getProperty("adMax"));
            upTime = Double.parseDouble(props.getProperty("upTime"));
            lowTime = Double.parseDouble(props.getProperty("lowTime"));
            downTime = Double.parseDouble(props.getProperty("downTime"));
            highTime = Double.parseDouble(props.getProperty("highTime", "0"));
            debugAD = Boolean.parseBoolean(props.getProperty("debugAD", "false"));
            System.out.println(">> Dynamic AD habilitado (highTime=" + highTime + ")");
        } else {
            System.out.println(">> Usando AD fixo = " + ad);
        }
    }

    private static void exec(int numDrones, BufferedWriter writer) {
        CountDownLatch latch = new CountDownLatch(1);
        Utils.setLatch(latch);

        // agendador e fila de pacotes
        SimScheduler scheduler = new SimScheduler();
        scheduler.simulationTime = simulationTime;
        int initialFleet = executionMode == SimScheduler.ExecutionMode.STATIC ? numDrones : minimoNumDrones;
        scheduler.configure(executionMode, rechargeTime, displacementTime, dtTargetMmt, dtIdleReserve, dtDecisionInterval, initialFleet, initialFleet);
        Queue<Package> packageQueue = new ConcurrentLinkedQueue<>();
        Utils.scheduler = scheduler;
        Utils.clearVariables();

        Utils.packageQueue = packageQueue;

        // --- Geração contínua de pacotes até simulationTime ---
        double tempoTotal = 0.0;
        int id = 0;
        while (tempoTotal < simulationTime) {
            double adParaGeracao = dynamicAD ? getDynamicAD(tempoTotal) : ad;
            //double adParaGeracao = ad;
            double atraso = Utils.generateExponential(adParaGeracao);

            tempoTotal += atraso;
            double chegada = tempoTotal;
            final int packageId = id++;

            scheduler.schedule("source", "AD", atraso, chegada, () -> {
                Utils.generatedPackagesNumber++;
                Package p = new Package(packageId, scheduler.now(), atraso);
                if (packageQueue.size() < maxQueueSize) {
                    packageQueue.add(p);
                    Utils.allocatedInQueuePackagesNumber++;
                    scheduler.dispatchAvailable();
                } else {
                    Utils.dropedPackagesNumber++;
                }

            });

            if (debugAD && id % 500 == 0) {
                System.out.printf(Locale.US, "[t=%.2f] debug gen: AD(geração)=%.3f%n", chegada, adParaGeracao);
            }
        }

        // --- Inicializa drones e registra callback de entrega ---
        for (int d = 0; d < initialFleet; d++) {
            Drone drone = new Drone(d, false, DroneState.AVAILABLE, packageQueue, scheduler);
            Utils.listaDeDrones.add(drone);

            // Configura callback para gravar no CSV quando o drone finalizar a entrega
            drone.setOnDelivery((missionTime) -> {
                double currentMMT = Utils.getLastMissionTime();   // ✅ usa valor real da missão
                double adAtual =  drone.lastAdExp; //dynamicAD ? getDynamicAD(scheduler.now()) : ad;
                try {
                    writer.write(String.format(Locale.US, "%.2f;%.4f;%.4f%n",
                            scheduler.now(), adAtual, currentMMT));
                    writer.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            // agenda a primeira tentativa de pickup no tempo 0 (cada drone)
            scheduler.schedule("Drone" + drone.id,"ToBase", 0, 0, drone::tryPickup);
        }

        // Define quantos drones estão disponiveis inicialmente
        for (int i = 0; i < Utils.numDronesAtivos; i++) {
            // Obtém o drone na posição i
            Drone drone = Utils.listaDeDrones.get(i);

            // Define o atributo ativo como true
            drone.state = DroneState.AVAILABLE;
        }

        // --- Agenda evento de término em simulationTime ---
        scheduler.schedule("simulation", "END", 1, simulationTime, () -> {
            System.out.println(">> Evento de término processado pelo scheduler (tempo simulado atingido).");
            // printResults chama Utils.printResults (imprime métricas)
            Utils.printResults();
            // sinaliza main thread
            latch.countDown();
        });

        // --- Executa o scheduler em thread separada (para não bloquear a thread principal) ---
        Thread schedulerThread = new Thread(() -> {
            try {
                scheduler.run();
            } catch (Throwable t) {
                t.printStackTrace();
            }
        }, "SimScheduler-Thread");
        schedulerThread.start();

        // --- Aguarda o latch com timeout de segurança ---
        try {
            long waitMillis = (long) (simulationTime * AWAIT_TIMEOUT_MULTIPLIER);
            boolean signaled = latch.await(waitMillis, TimeUnit.MILLISECONDS);
            if (!signaled) {
                System.err.println("⚠ Timeout aguardando término da simulação. Irá continuar e imprimir resultados parciais.");
                // tenta garantir que resultados sejam impressos
                Utils.printResults();
            } else {
                System.out.println(">> Latch sinalizado — simulação finalizada (ou evento de término processado).");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Thread aguardando latch foi interrompida.");
        }

        // --- Tentativa de encerrar thread do scheduler limpa/educadamente ---
        if (schedulerThread.isAlive()) {
            try {
                // se SimScheduler tiver método stop()/shutdown() preferível chamá-lo aqui.
                // Neste código não sabemos, então interrompemos a thread como tentativa:
                schedulerThread.interrupt();
                schedulerThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Após o await (ou timeout), garantimos que os resultados finais estão impressos e liberamos recursos.
    }

    private static double getDynamicAD(double currentTime) {
        double cycleTime = upTime + highTime + downTime + lowTime;
        double t = currentTime % cycleTime;

        if (t < upTime) {
            // subida: adMin -> adMax
            return adMin + (adMax - adMin) * (t / upTime);

        } else if (t < upTime + highTime) {
            // patamar no máximo
            return adMax;

        } else if (t < upTime + highTime + downTime) {
            // descida: adMax -> adMin
            double elapsed = t - (upTime + highTime);
            return adMax - (adMax - adMin) * (elapsed / downTime);

        } else {
            // patamar no mínimo
            return adMin;
        }
    }
}
