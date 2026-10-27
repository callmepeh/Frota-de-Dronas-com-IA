import org.ufpi.mercurycall.solvers.NonExactResult;
import org.ufpi.mercurycall.solvers.Parameter;
import org.ufpi.mercurycall.solvers.SPNsolver;
import org.ufpi.mercurycall.util.facade.utils.FacadeUtil;

import java.io.*;
import java.util.*;

public class BatchSPNSolver_BACKUP {



	public static void main(String[] args) throws Exception {
		FacadeUtil.disableWarning();

		List<SystemState> instances = loadFromFile("system_state_log.csv");

		SPNsolver spnSolver = new SPNsolver("modelo/model_drones_validation_DT.xml"); //model_drones_validation_DT_infiniteserver


		for (SystemState modelParameters : instances) {
			runSPNSolver(spnSolver, modelParameters);
		}
	}

	static class SystemState {
		double time;
		double ad;
		double queueU;
		int pousoDecola1Size;
		double pousoDecola1;
		int voo1Size;
		double voo1;
		int pousoDecola2Size;
		double pousoDecola2;
		int pousoDecola3Size;
		double pousoDecola3;
		int voo2Size;
		double voo2;
		double pousoDecola4;
		double mmt;
		double inMissionDrones;

		@Override
		public String toString() {
			return String.format(Locale.US,
					"t=%.2f AD=%.3f QUEUE_U=%.2f MMT=%.3f",
					time, ad, queueU, mmt);
		}
	}


	public static List<SystemState> loadFromFile(String filename) {
		List<SystemState> states = new ArrayList<>();

		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
			String line;
			boolean first = true;

			while ((line = reader.readLine()) != null) {
				// pula cabeçalho
				if (first) {
					first = false;
					continue;
				}

				// separa por ";"
				String[] parts = line.split(";");
				if (parts.length < 16) continue; // valida integridade

				SystemState s = new SystemState();
				s.time = Double.parseDouble(parts[0]);
				s.ad = Double.parseDouble(parts[1]);
				s.queueU = Double.parseDouble(parts[2]);
				s.pousoDecola1Size = Integer.parseInt(parts[3]);
				s.pousoDecola1 = Double.parseDouble(parts[4]);
				s.voo1Size = Integer.parseInt(parts[5]);
				s.voo1 = Double.parseDouble(parts[6]);
				s.pousoDecola2Size = Integer.parseInt(parts[7]);
				s.pousoDecola2 = Double.parseDouble(parts[8]);
				s.pousoDecola3Size = Integer.parseInt(parts[9]);
				s.pousoDecola3 = Double.parseDouble(parts[10]);
				s.voo2Size = Integer.parseInt(parts[11]);
				s.voo2 = Double.parseDouble(parts[12]);
				s.pousoDecola4 = Double.parseDouble(parts[13]);
				s.mmt = Double.parseDouble(parts[14]);
				s.inMissionDrones = Double.parseDouble(parts[15]);

				states.add(s);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

		return states;
	}


	private static void runSPNSolver(SPNsolver spnSolver, SystemState modelParameters) throws Exception {
		writeMMTSystem(String.valueOf(modelParameters.mmt));


		int numDrones = 6;
		int tamDeposito = 1000;
		double totalDronesUtilizados = modelParameters.pousoDecola1Size + modelParameters.voo1Size + modelParameters.pousoDecola2Size + modelParameters.pousoDecola3Size + modelParameters.voo2Size;
		double A = Math.round(percentageOfTotal(modelParameters.pousoDecola1Size, totalDronesUtilizados) * modelParameters.inMissionDrones);
		double B = Math.round(percentageOfTotal(modelParameters.voo1Size, totalDronesUtilizados) * modelParameters.inMissionDrones);
		double C = Math.round(percentageOfTotal(modelParameters.pousoDecola2Size, totalDronesUtilizados) * modelParameters.inMissionDrones);
		double D = Math.round(percentageOfTotal(modelParameters.pousoDecola3Size, totalDronesUtilizados) * modelParameters.inMissionDrones);
		double E = Math.round(percentageOfTotal(modelParameters.voo2Size, totalDronesUtilizados) * modelParameters.inMissionDrones);

		double drones = numDrones - (A + B + C + D + E);

		double VOO = (modelParameters.voo1 + modelParameters.voo2) / 2;
		double POUSO_DECOL = (modelParameters.pousoDecola1 + modelParameters.pousoDecola2 + modelParameters.pousoDecola3 + modelParameters.pousoDecola4) / 4;


		spnSolver.changeParameter("AD", String.valueOf(modelParameters.ad));

		spnSolver.changeParameter("DEPOSITO", String.valueOf(Math.abs(tamDeposito - Math.round(modelParameters.queueU * tamDeposito / 100))));
		spnSolver.changeParameter("PACOTES", String.valueOf(Math.round(modelParameters.queueU  * tamDeposito / 100)));

		spnSolver.changeParameter("DRONES", String.valueOf(drones));
		spnSolver.changeParameter("A", String.valueOf(A));
		spnSolver.changeParameter("B", String.valueOf(B));
		spnSolver.changeParameter("C", String.valueOf(C));
		spnSolver.changeParameter("D", String.valueOf(D));
		spnSolver.changeParameter("E", String.valueOf(E));
		// Calcular media de pousos e decolagens
		spnSolver.changeParameter("VOO", String.valueOf(VOO));
		spnSolver.changeParameter("POUSO_DECOL", String.valueOf(POUSO_DECOL));


		for (Parameter parameter : spnSolver.getParameters()) {
			System.out.println(parameter.getName() + " " + parameter.getValue());
		}


		HashMap<String, NonExactResult> NRemodelResult=null;

		NRemodelResult=spnSolver.stationarySimulationSolve(0, 0.1);

		double mmt = 0;
		if(NRemodelResult!=null){
			for (Map.Entry<String, NonExactResult> entry : NRemodelResult.entrySet()) {
				String key = entry.getKey();
				NonExactResult result = entry.getValue();
				
				if(key.contains("MMT"))
					mmt = result.getMean();
			}
		}


		System.err.println(mmt);
		writeMMTModel(String.valueOf(mmt));
	}

	static double percentageOfTotal(double value, double total){
		return value/total;
	}


	private static void writeMMTModel(String value) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter("MODEL_MRT.txt", true))) {
			bw.write(value);
			bw.newLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void writeMMTSystem(String value) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter("SYSTEM_MRT.txt", true))) {
			bw.write(value);
			bw.newLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
