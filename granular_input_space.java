import java.util.Arrays;
import java.util.Date;
import java.util.Random;



public class granular_input_space {

	public String dirName = "\\..energy\\";
	public String dataFileName = "energy.txt";
	
	public String configureFileName = "NN_info_inputs_nodes.txt";
	public int c_value = 3;
	
	public double membership_matrix[][][];  //每个维度上有一个二维的membership矩阵，每一列代表当前数据对于每一个模糊集的隶属度
	
	public double[][] dataMatrix;
	public final double PI = 3.14;
	public int rows; // rows of data
	public int dimension; //dimensionality of input data;
	public int num_of_layers; //number of neural network layers
	public int[] nodes_per_layer; // number of nodes in each layer 
	public double[][][] link_weights; //weights of links
	public double[][] bias;   //bias in each layer
	public double[][] weights_of_fuzzy_sets; 
	public double[][] dataMatrix_upper_bounds;
	public double[][] dataMatrix_lower_bounds;

	Date now = new Date();
	long nowLong = now.getTime();
	Random randomno = new Random();
	
	public result_spec_incl_pair calculate_Y_output() {
		dataMatrix_upper_bounds = new double[rows][dimension];
		dataMatrix_lower_bounds = new double[rows][dimension];
		double[][] all_output_interval = new double[rows][];
		
		for(int row = 0; row < dataMatrix.length; row++) {
			//对输入进行粒度化，将x(x1, x2, ..., xn)变换成区间X([x1-,x1+], [x2-, x2+],..., [xn-, xn+])
			for(int cur_dimension = 0; cur_dimension < dimension; cur_dimension++) {
				double cur_dimension_x = dataMatrix[row][cur_dimension];
				double epsilon = 0;
				for(int i = 0; i < c_value; i++) {
					epsilon += membership_matrix[cur_dimension][row][i] * weights_of_fuzzy_sets[cur_dimension][i];
				}
				//System.out.printf("dimension = %d, epsilon = %.2f \n", cur_dimension, epsilon);
					
				double v1 = cur_dimension_x / (1 + epsilon);
				double v2 = cur_dimension_x * (1 + epsilon);
				dataMatrix_upper_bounds[row][cur_dimension] = max(v1, v2);
				dataMatrix_lower_bounds[row][cur_dimension] = min(v1, v2);
//				System.out.printf("epsilon = %.2f value[%d %d]=[%.2f %.2f]\n", epsilon,row, cur_dimension, dataMatrix_lower_bounds[row][cur_dimension], dataMatrix_upper_bounds[row][cur_dimension]);
			}
			
			//如果输入为x，那么输出是一个单个的数值y
			double y_output = calculate_ouput_for_vector(dataMatrix[row]);
			//X对应的输出Y也不是单个的数值，而是区间形式
			all_output_interval[row] = calculate_ouput_for_granular(dataMatrix_lower_bounds[row], dataMatrix_upper_bounds[row]);
			//y应该位于Y区间内，否则表明程序有错
			assert((all_output_interval[row][0] < y_output) && (all_output_interval[row][1] > y_output));
//			System.out.printf("single value output: %.2f -- interval output: [%.2f %.2f]\n", y_output, all_output_interval[row][0], all_output_interval[row][1]);
		}
		
		//计算y的最大值和最小值，然后分别计算每个输出的具体性指标
		double max_y_output = Double.MIN_VALUE;
		double min_y_output = Double.MAX_VALUE;
		int included_y_by_Y = 0;
		for(int row = 0; row < dataMatrix.length; row++) {
			
//			System.out.printf("%.2f -- %.2f -- %.2f\n", all_output_interval[row][0], dataMatrix[row][dimension], all_output_interval[row][1]);
			if((all_output_interval[row][1] > dataMatrix[row][dimension]) && (all_output_interval[row][0]  < dataMatrix[row][dimension])) {
//				System.out.println("included=================");
				included_y_by_Y++;
			}
			max_y_output = max_y_output > all_output_interval[row][1] ? max_y_output : all_output_interval[row][1];
			min_y_output = min_y_output < all_output_interval[row][0] ? min_y_output : all_output_interval[row][0];
		}
		
		double output_range = max_y_output - min_y_output;
		double[] specificities = new double[rows];
		
		for(int row = 0; row < dataMatrix.length; row++) {
			specificities[row] = max(0, 1 - (all_output_interval[row][1] - all_output_interval[row][0]) / output_range);
//			System.out.printf("%.2f ", specificities[row]);

			assert(specificities[row] >= 0 && specificities[row] <= 1);
		}
		
		double overall_specificity = myAdd(specificities);
		
//		System.out.printf("min_y_output = %.2f, max_y_output = %.2f\n", min_y_output, max_y_output);
		
		return new result_spec_incl_pair(overall_specificity / dataMatrix.length, included_y_by_Y / (double)dataMatrix.length);
	}
	
	//初始化权重矩阵，权重矩阵之和为target_epsilon_each_fuzzy_set，有两种计算方式，TBD
	public void initialize_weights_of_fuzzy_sets(double target_epsilon_each_fuzzy_set) {
		double target_weight = dimension * target_epsilon_each_fuzzy_set;
		//double target_weight = target_epsilon_each_fuzzy_set;

		weights_of_fuzzy_sets = new double[dimension][c_value];
		double sum_of_weights_on_all_dimension = 0;
		for(int i = 0; i < dimension; i++) {
			double sum_on_current_dimension = 0;
			for(int j = 0; j < c_value; j++) {
				weights_of_fuzzy_sets[i][j] = randomno.nextDouble();
				sum_on_current_dimension += weights_of_fuzzy_sets[i][j] ;
			}
			
			sum_of_weights_on_all_dimension += sum_on_current_dimension / c_value;
		}
		
		for(int i = 0; i < dimension; i++) {
			for(int j = 0; j < c_value; j++) {
				weights_of_fuzzy_sets[i][j]  = weights_of_fuzzy_sets[i][j]  * target_weight / sum_of_weights_on_all_dimension;
			}
		}
/*		
 * 		第一次试验使用的分配方式，及一个变量在各维度上的信息粒度之和等于n*epsilon
        weights_of_fuzzy_sets = new double[dimension][c_value];
		for(int i = 0; i < c_value; i++) {
			boolean condition_satisfied = false;
			int try_round = 0;
			while(condition_satisfied == false) {
				try_round++;
				double sum_of_weights_on_cur_dimension = 0;
				for(int j = 0; j < dimension; j++) {
					weights_of_fuzzy_sets[j][i] = randomno.nextDouble();
					sum_of_weights_on_cur_dimension += weights_of_fuzzy_sets[j][i];
				}
				
				for(int j = 0; j < dimension; j++) {
					weights_of_fuzzy_sets[j][i] = weights_of_fuzzy_sets[j][i] * target_weight / sum_of_weights_on_cur_dimension;
				}
	
				condition_satisfied = true;
				for(int j = 0; j < dimension; j++) {
					if (weights_of_fuzzy_sets[j][i] > 1 || weights_of_fuzzy_sets[j][i] < 0) {
						condition_satisfied = false;
					}
				}
			}
*/			
//			for(int j = 0; j < c_value; j++) {
//				System.out.printf("%.2f ", weights_of_fuzzy_sets[i][j]);
//			}
//			System.out.printf("try_round = %d --sum of weights on current dimension = %.2f\n", try_round, myAdd(weights_of_fuzzy_sets[i]));
//		}

//		System.out.println("---------------------");
//		for(int i = 0; i < dimension; i++) {
//			for(int j = 0; j < c_value; j++) {
//				if(i == 0) {weights_of_fuzzy_sets[i][0] = 0.42;weights_of_fuzzy_sets[i][1] = 0.42;weights_of_fuzzy_sets[i][2] = 0.42;}
//				if(i == 1) {weights_of_fuzzy_sets[i][0] = 0.20;weights_of_fuzzy_sets[i][1] = 0.20;weights_of_fuzzy_sets[i][2] = 0.20;}
//				System.out.printf("%.2f ", weights_of_fuzzy_sets[i][j]);
//			}
//			System.out.println();
//		}

	}
	
	public void normalize_weights_of_fuzzy_sets(double[][] weights_of_fuzzy_sets, double target_weight) {
		for(int i = 0; i < dimension; i++) {
			double sum_of_weights_on_cur_dimension = 0;
			for(int j = 0; j < c_value; j++) {
				sum_of_weights_on_cur_dimension += weights_of_fuzzy_sets[i][j];
			}
			
			for(int j = 0; j < c_value; j++) {
				weights_of_fuzzy_sets[i][j] = weights_of_fuzzy_sets[i][j] * target_weight / sum_of_weights_on_cur_dimension;
			}
			System.out.println(Arrays.toString(weights_of_fuzzy_sets[i]));
			System.out.printf("sum of weights on current dimension = %.2f\n", myAdd(weights_of_fuzzy_sets[i]));
		}
	}
	
	public void initialize_membership_matrix() {
		membership_matrix = new double[dimension][][];
		assert(rows == dataMatrix.length);
		
		for(int cur_dimension = 0; cur_dimension < dimension; cur_dimension++) {
//			System.out.printf("----------------cur_dimension = %d\n", cur_dimension);

			membership_matrix[cur_dimension] = new double[rows][c_value];
			double maxV = Double.MIN_VALUE, minV = Double.MAX_VALUE;
			
			for(int row = 0; row < dataMatrix.length; row++) {
				if(dataMatrix[row][cur_dimension] > maxV) {
					maxV = dataMatrix[row][cur_dimension];
				}
				
				if(dataMatrix[row][cur_dimension] < minV) {
					minV = dataMatrix[row][cur_dimension];
				}
			}
//			System.out.printf("minV = %.2f, maxV = %.2f\n", minV, maxV);
			
			//每个维度上的模糊函数的中心点
			double[] modal_values = new double[c_value];
			for(int i = 0; i < c_value; i++) {
				modal_values[i] = minV + (maxV - minV) / (c_value - 1) * i;
			}
			//高斯分布的方差
			double variance = (maxV - minV) / (c_value - 1);

			for(int row = 0; row < rows; row++) {
				double cur_x = dataMatrix[row][dimension];
				for(int i = 0; i < c_value; i++) {
					membership_matrix[cur_dimension][row][i] = Triangular(cur_x, modal_values[i], variance);
//					System.out.printf("cur_x = %.2f, modal_value[%d] = %.2f variance = %.2f Triangular_value = %.2f\n", 
//							cur_x, i, modal_values[i], variance, membership_matrix[cur_dimension][row][i]);
				}
//				System.out.println(Arrays.toString(membership_matrix[cur_dimension][row]));
//				System.out.println(myAdd(membership_matrix[cur_dimension][row]));
			}
			
//			System.out.println(Arrays.toString(modal_values));
		} // end of loop for dimension
	}
	
	public double myAdd(double[] v_array) {
		double answer = 0;
		
		for(int i = 0; i < v_array.length; i++) {
			answer += v_array[i];
		}
		
		return answer;
	}
	public double Triangular(double x, double modal_value, double variance) {
		double answer = 0;
		if((x < modal_value - variance) || (x > modal_value + variance)) {
			answer = 0;
		} else {
			answer = 1 - Math.abs(modal_value - x) / variance;
		}
		return answer;
	}
	public double Gaussian(double x, double modal_value, double variance) {
		double answer = 0;
		
		answer = (1.0 / (variance * Math.sqrt(2 * PI))) 
				* Math.exp((-1.0)  * ((x - modal_value) * (x - modal_value)) / (2 * variance * variance));
		
		return answer;
	}
	
	
	public granular_input_space() {
		String dataFilePath = dirName + dataFileName;
		dataMatrix = Utils.readDoubleFromTxtToMatrix(dataFilePath, " ");
		System.out.println("dataMatrix.size=["+dataMatrix.length + ", " + dataMatrix[0].length + "]");
		
		//将数据归一化到[-1, 1]区间
		Utils.normalizeDataMatrix(dataMatrix, -1, 1);
	
		//Utils.printDataMatrix(dataMatrix);
	}
	
	//这些config文件是由matlab程序生成的
	public void read_configure_file() {
		//读取配置文件
		String configureFile = dirName + configureFileName;
		int[][] configureData = Utils.readIntFromTxtToMatrix(configureFile);
		assert(configureData.length == 1);
		
		//config文件的数值分别表示：数据的行数，输入数据的维度，神经网络的层数，每一层的节点数
		rows = (int) configureData[0][0];
		dimension = (int) configureData[0][1];
		num_of_layers = (int) configureData[0][2];
		nodes_per_layer = new int[num_of_layers];
		
		for(int i = 0; i < num_of_layers; i++) {
			nodes_per_layer[i] = (int) configureData[0][i + 3];
		}
		
//		System.out.println("dimension = " + dimension);
//		System.out.println("num_of_layers = " + num_of_layers);
//		for(int i = 0; i < num_of_layers; i++) {
//			System.out.printf("num_of_layers[%d] = %d\n", i, nodes_per_layer[i]);
//		}
	}
	
	//根据matlab程序训练的神经网络，来初始化各参数变量矩阵
	public void init_link_weights_bias() {
		System.out.println("init_link_weights: num_of_layers = " + num_of_layers);

		link_weights = new double[num_of_layers][][];
		bias = new double[num_of_layers][];
		
//		link_weights[0] = new double[dimension][nodes_per_layer[0]]; //input layer
//		for(int i = 0; i < num_of_layers - 1; i++) {
//			link_weights[i + 1] = new double[nodes_per_layer[i]][nodes_per_layer[i + 1]];
//		}

		//read weights for input layer
		String inputWeightsFile = "inputWeights.txt";
		String biasFile = "inputbias.txt";
		String weightsFilePath = dirName + inputWeightsFile;
		String biasFilePath = dirName + biasFile;
		link_weights[0] = Utils.readDoubleFromTxtToMatrix(weightsFilePath, "\t");
		bias[0] = Utils.readDoubleFromTxtToMatrix(biasFilePath, "\t")[0];
		
		if(num_of_layers >= 2) {
			inputWeightsFile = "layerWeights2.txt";
			biasFile = "layerbias2.txt";

			weightsFilePath = dirName + inputWeightsFile;
			biasFilePath = dirName + biasFile;
			
			link_weights[1] = Utils.readDoubleFromTxtToMatrix(weightsFilePath, "\t");
			bias[1] = Utils.readDoubleFromTxtToMatrix(biasFilePath, "\t")[0];
		}
		
		if(num_of_layers >= 3) {
			inputWeightsFile = "layerWeights3.txt";
			biasFile = "layerbias3.txt";

			weightsFilePath = dirName + inputWeightsFile;
			biasFilePath = dirName + biasFile;
			
			link_weights[2] = Utils.readDoubleFromTxtToMatrix(weightsFilePath, "\t");
			bias[2] = Utils.readDoubleFromTxtToMatrix(biasFilePath, "\t")[0];
		}
		
		if(num_of_layers >= 4) {
			inputWeightsFile = "layerWeights4.txt";
			biasFile = "layerbias4.txt";

			weightsFilePath = dirName + inputWeightsFile;
			biasFilePath = dirName + biasFile;
			
			link_weights[3] = Utils.readDoubleFromTxtToMatrix(weightsFilePath, "\t");
			bias[3] = Utils.readDoubleFromTxtToMatrix(biasFilePath, "\t")[0];
		}
		

		for(int i = 0; i < num_of_layers; i++) {
			System.out.printf("i = %d\n", i);
			System.out.printf("num of links of index [%d] = %d\n", i, link_weights[i].length * link_weights[i][0].length);
		}
	}
	

	public double calculate_ouput_for_vector(double[] x) {
		double answer = 0;
		assert(x.length == dimension);
		
		double[] output_of_prev_layer = x;
		for(int layer = 0; layer < num_of_layers - 1; layer++) {
			double[] output_of_current_layer = new double[nodes_per_layer[layer]];
			
			for(int j = 0; j < nodes_per_layer[layer]; j++) {
				output_of_current_layer[j] = 0;
				
				int num_of_nodes_prev_layer = (layer>0)? nodes_per_layer[layer - 1] : dimension;
				assert(num_of_nodes_prev_layer == output_of_prev_layer.length);
				
				for(int i = 0; i < num_of_nodes_prev_layer; i++) {
					output_of_current_layer[j] +=  link_weights[layer][i][j] * output_of_prev_layer[i];
				}
				
				output_of_current_layer[j] += bias[layer][j];
				//fire function
				output_of_current_layer[j] = tansig(output_of_current_layer[j]);
			}
			
			output_of_prev_layer = output_of_current_layer;
		}
		
		//计算最后一层的输出
		for(int i = 0; i < nodes_per_layer[num_of_layers - 2]; i++) {
			answer += link_weights[num_of_layers - 1][i][0] * output_of_prev_layer[i];
		}
		answer += bias[num_of_layers - 1][0];
		
		return answer;
	}
	
    public double logsig(double x)
    {
    	return  1.0 / (1 + Math.exp(-1 * x));
    }
    
	public double tansig(double x) {
		return 2 * logsig(2 * x) - 1;
	}
	
	
	public double max(double a, double b) {
		return a > b ? a : b;
	}
	
	public double min(double a, double b) {
		return a < b ? a : b;
	}
	
	public double[] calculate_ouput_for_granular(double[] lower_bounds, double[] upper_bounds) {
		double[] answer = new double[2];
		assert(lower_bounds.length == dimension);
		assert(upper_bounds.length == dimension);
	
		double[] lower_output_of_prev_layer = lower_bounds;
		double[] upper_output_of_prev_layer = upper_bounds;

		for(int layer = 0; layer < num_of_layers - 1; layer++) {
			double[] lower_output_of_current_layer = new double[nodes_per_layer[layer]];
			double[] upper_output_of_current_layer = new double[nodes_per_layer[layer]];

			for(int j = 0; j < nodes_per_layer[layer]; j++) {
				lower_output_of_current_layer[j] = 0;
				upper_output_of_current_layer[j] = 0;
				
				int num_of_nodes_prev_layer = (layer>0)? nodes_per_layer[layer - 1] : dimension;
				assert(num_of_nodes_prev_layer == lower_output_of_current_layer.length);
				assert(num_of_nodes_prev_layer == upper_output_of_current_layer.length);

				for(int i = 0; i < num_of_nodes_prev_layer; i++) {
					double v1 = link_weights[layer][i][j] * lower_output_of_prev_layer[i];
					double v2 = link_weights[layer][i][j] * upper_output_of_prev_layer[i];
					double lower_value = min(v1, v2);
					double upper_value = max(v1, v2);
					
					lower_output_of_current_layer[j] +=  lower_value;
					upper_output_of_current_layer[j] +=  upper_value;
				}
				
				lower_output_of_current_layer[j] += bias[layer][j];
				upper_output_of_current_layer[j] += bias[layer][j];

				//fire function
				lower_output_of_current_layer[j] = tansig(lower_output_of_current_layer[j]);
				upper_output_of_current_layer[j] = tansig(upper_output_of_current_layer[j]);
			}
			
			lower_output_of_prev_layer = lower_output_of_current_layer;
			upper_output_of_prev_layer = upper_output_of_current_layer;
		}
		
		//计算最后一层的输出
		for(int i = 0; i < nodes_per_layer[num_of_layers - 2]; i++) {
			double v1 = link_weights[num_of_layers - 1][i][0] * lower_output_of_prev_layer[i];
			double v2 = link_weights[num_of_layers - 1][i][0] * upper_output_of_prev_layer[i];
			answer[0] +=  min(v1, v2);
			answer[1] +=  max(v1, v2);
		}
		answer[0] += bias[num_of_layers - 1][0];
		answer[1] += bias[num_of_layers - 1][0];

		return answer;
	}

	


	
}




///*================测试代码================*/		
//double[][] xx = {{0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3}, {0.4, 0.4}, {0.5, 0.5}};
//for(int i = 0; i < xx.length; i++) {
//	double[] x = xx[i];
//	double y_output = test_instance.calculate_ouput_for_vector(x);
//	System.out.printf("%s --output-- %.6f\n", Arrays.toString(x), y_output);
//}
//
//double[][] lower_x = {{0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3}, {0.4, 0.4}, {0.5, 0.5}};
//double[][] upper_x = {{0.1, 0.1}, {0.3, 0.3}, {0.4, 0.4}, {0.5, 0.5}, {0.6, 0.6}};
//for(int i = 0; i < xx.length; i++) {
//	double[] lower_bounds = lower_x[i];
//	double[] upper_bounds = upper_x[i];
//	double[] Y_OUTPUT = test_instance.calculate_ouput_for_granular(lower_bounds, upper_bounds);
//	System.out.printf("%s ~ %s --output-- %s\n", Arrays.toString(lower_bounds), Arrays.toString(upper_bounds), Arrays.toString(Y_OUTPUT));
//}
///*================测试代码================*/	