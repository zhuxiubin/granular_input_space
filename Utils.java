
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
	public static double[][] readDoubleFromTxtToMatrix(String path, String sep) {
		  File file = new File(path);
		  List<String> list = new ArrayList<String>();
		  double[][] dataMatrix = null;
		  try {
			  BufferedReader bw = new BufferedReader(new FileReader(file));
			  String line = null;
			  //将一行行的数据添加到list
			  while(((line = bw.readLine()) != null) && (!line.trim().equals(""))) {
				  list.add(line);
			  }
			  bw.close();
		  } catch (IOException e) {
			  e.printStackTrace();
		  }

		  int numPerLine = ((String)list.get(0)).replaceAll("  ", " ").replaceAll("  ", " ").replaceAll("  ", " ").replaceAll("  ", " ").split(sep).length;
		  dataMatrix = new double[list.size()][numPerLine];
System.out.printf("numPerLine = %d, list.size(, args) = %d\n", numPerLine, list.size());

		  for(int i=0;i<list.size();i++){
			  	String Line = (String) list.get(i);
			  	Line = Line.replaceAll("  ", " ").replaceAll("  ", " ").replaceAll("  ", " ").replaceAll("  ", " ");
		  		String[] singleNumArray = Line.split(sep);
		  		
		  		System.out.println(Arrays.toString(singleNumArray));
		  		
			  	for(int j = 0; j < numPerLine; j++) {
			  		dataMatrix[i][j] = Double.parseDouble(singleNumArray[j]);
			  	}
		  }
		  return dataMatrix;
	}

	public static int[][] readIntFromTxtToMatrix(String path) {
		  File file = new File(path);
		  List<String> list = new ArrayList<String>();
		  int[][] dataMatrix = null;
		  try {
			  BufferedReader bw = new BufferedReader(new FileReader(file));
			  String line = null;
			  //将一行行的数据添加到list
			  while(((line = bw.readLine()) != null) && (!line.trim().equals(""))) {
				  list.add(line);
			  }
			  bw.close();
		  } catch (IOException e) {
			  e.printStackTrace();
		  }

		  int numPerLine = ((String)list.get(0)).split("\t").length;
		  dataMatrix = new int[list.size()][numPerLine];
		  for(int i=0;i<list.size();i++){
			  	String Line = (String) list.get(i);
		  		String[] singleNumArray = Line.split("\t");
		  		
			  	for(int j = 0; j < numPerLine; j++) {
			  		//System.out.println(singleNumArray[j]);
			  		dataMatrix[i][j] = Integer.parseInt(singleNumArray[j]);
			  	}
		  }
		  return dataMatrix;
	}

	public static void normalizeDataMatrix(double[][] dataMatrix, double lower_bound, double upper_bound) {
		
		for(int  col = 0; col < dataMatrix[0].length; col++) {
			double minV = Double.MAX_VALUE;
			double maxV = Double.MIN_VALUE;
			
			for(int row = 0; row < dataMatrix.length; row++) {
				if(dataMatrix[row][col] > maxV) {
					maxV = dataMatrix[row][col];
				}
				
				if(dataMatrix[row][col] < minV) {
					minV = dataMatrix[row][col];
				}
			}

			for(int row = 0; row < dataMatrix.length; row++) {
				dataMatrix[row][col] = lower_bound + (upper_bound - lower_bound) * (dataMatrix[row][col] - minV) / (maxV - minV);
			}
		}
	}
	
	public static void printDataMatrix(double[][] dataMatrix) {
		  for(int i = 0;i < dataMatrix.length; i++){
			  	for(int j = 0; j < dataMatrix[0].length; j++) {
			  		System.out.printf("%.2f ", dataMatrix[i][j]);
			  	}
			  	System.out.println("");
		  }
	}
}
