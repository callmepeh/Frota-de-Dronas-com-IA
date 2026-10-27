import org.ufpi.mercurycall.solvers.NonExactResult;
import org.ufpi.mercurycall.solvers.Parameter;
import org.ufpi.mercurycall.solvers.SPNsolver;
import org.ufpi.mercurycall.util.facade.utils.FacadeUtil;

import java.io.*;
import java.util.*;

public class BatchSPNSolver {



	public static void main(String[] args) throws Exception {
		FacadeUtil.disableWarning();

		List<SystemState> instances = loadFromFile("system_state_log.csv");

		SPNsolver spnSolver = new SPNsolver("modelo/model_drones_validation_DT.xml"); //model_drones_validation_DT_infiniteserver


		for (SystemState modelParameters : instances) {
			if(modelParameters.mmt > 0 && modelParameters.ad > 0)
				runSPNSolver(spnSolver, modelParameters);
		}
	}

	static class SystemState {
		double time;
		double ad;
		double queue;
		double pousoDecola;
		double voo;
		double mmt;
		double tp;

		// Novos valores
		double A;
		double B;
		double C;
		double D;
		double E;

		@Override
		public String toString() {
			return String.format(Locale.US,
					"t=%.2f AD=%.3f QUEUE=%.3f MMT=%.3f TP=%.3f A=%.3f B=%.3f C=%.3f D=%.3f E=%.3f",
					time, ad, queue, mmt, tp, A, B, C, D, E);
		}
	}


	public static List<SystemState> loadFromFile(String filename) {
		List<SystemState> states = new ArrayList<>();

		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
			String line;
			boolean first = true;

			while ((line = reader.readLine()) != null) {

				if (first) {
					first = false;
					continue; // pula cabeçalho
				}

				String[] parts = line.split(";");
				if (parts.length < 12) continue; // agora exige 12 colunas

				SystemState s = new SystemState();
				s.time = Double.parseDouble(parts[0]);
				s.ad = Double.parseDouble(parts[1]);
				s.queue = Double.parseDouble(parts[2]);
				s.pousoDecola = Double.parseDouble(parts[3]);
				s.voo = Double.parseDouble(parts[4]);
				s.mmt = Double.parseDouble(parts[5]);
				s.tp = Double.parseDouble(parts[6]);

				s.A = Double.parseDouble(parts[7]);
				s.B = Double.parseDouble(parts[8]);
				s.C = Double.parseDouble(parts[9]);
				s.D = Double.parseDouble(parts[10]);
				s.E = Double.parseDouble(parts[11]);

				states.add(s);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

		return states;
	}


	private static void runSPNSolver(SPNsolver spnSolver, SystemState modelParameters) throws Exception {
		writeMMTSystem(String.valueOf(modelParameters.tp));


		int numDrones = 100;
		int tamDeposito = 1000;
		double A = modelParameters.A;
		double B = modelParameters.B;
		double C = modelParameters.C;
		double D = modelParameters.D;
		double E = modelParameters.E;

		double drones = numDrones - (A + B + C + D + E);

		double sumTokens = Math.round(modelParameters.queue) + (A + B + C + D + E);

		spnSolver.changeParameter("TOKENS", String.valueOf(sumTokens));

		spnSolver.changeParameter("AD", String.valueOf(modelParameters.ad));

		spnSolver.changeParameter("DEPOSITO", String.valueOf(Math.abs(tamDeposito - Math.round(modelParameters.queue))));
		spnSolver.changeParameter("PACOTES", String.valueOf(Math.round(modelParameters.queue)));

		spnSolver.changeParameter("DRONES", String.valueOf(drones));

		spnSolver.changeParameter("A", String.valueOf(A));
		spnSolver.changeParameter("B", String.valueOf(B));
		spnSolver.changeParameter("C", String.valueOf(C));
		spnSolver.changeParameter("D", String.valueOf(D));
		spnSolver.changeParameter("E", String.valueOf(E));

		// Calcular media de pousos e decolagens
		spnSolver.changeParameter("VOO", String.valueOf(modelParameters.voo));
		spnSolver.changeParameter("POUSO_DECOL", String.valueOf(modelParameters.pousoDecola));


		for (Parameter parameter : spnSolver.getParameters()) {
			System.out.println(parameter.getName() + " " + parameter.getValue());
		}

/*
		List<NonExactResult> total = spnSolver.transientSimulation(0.95, 0.1, 0, 1, 2);
		System.err.println(total);
		writeMMTModel(String.valueOf(total.get(0).getMean()));
*/

		HashMap<String, NonExactResult> NRemodelResult=null;

		NRemodelResult=spnSolver.stationarySimulationSolve(0, 0.1);

		double mmt = 0;
		if(NRemodelResult!=null){
			for (Map.Entry<String, NonExactResult> entry : NRemodelResult.entrySet()) {
				String key = entry.getKey();
				NonExactResult result = entry.getValue();
				
				if(key.contains("TP"))
					mmt = result.getMean();
			}
		}


		System.err.println(mmt);
		writeMMTModel(String.valueOf(mmt));


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
