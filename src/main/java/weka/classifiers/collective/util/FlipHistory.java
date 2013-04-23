/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * FlipHistory.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Utils;
  
/**
 * A flipping algorithm, that flips the label only if the previous and
 * the current (possibly new) label disagree more than a certain percentage.
 *
 * @author    Kurt Driessens (kurtd at cs dot waikato dot ac dot nz)
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public class FlipHistory
  implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = -4270199162924992735L;

  /** the sorted instances */
  protected Instance[] m_Instances = null;

  /** the last distribution */
  protected double[][] m_Last = null;

  /** the averaged distributions */
  protected double[][] m_Average = null;

  /** indicates how many distribution for a specific instance were added */
  protected int[] m_Count = null;

  /** singleton for comparing */
  protected InstanceComparator m_Comparator = new InstanceComparator(false);

  /** whether at least one distribution was added to the history */
  protected boolean m_HasHistory = false;

  /**
   * initializes the history
   * 
   * @param inst	the instance to initialize with
   */
  public FlipHistory(Instances inst) {
    int       i;
    
    // create arrays
    m_Instances = new Instance[inst.numInstances()];
    m_Last      = new double[inst.numInstances()][inst.numClasses()];
    m_Average   = new double[inst.numInstances()][inst.numClasses()];
    m_Count     = new int[inst.numInstances()];

    // sort
    for (i = 0; i < inst.numInstances(); i++)
      m_Instances[i] = (Instance) inst.instance(i).copy();
    Arrays.sort(m_Instances, m_Comparator);

    // init
    for (i = 0; i < m_Instances.length; i++) {
      m_Last[i][(int) m_Instances[i].classValue()]    = 1.0;
      m_Average[i][(int) m_Instances[i].classValue()] = 1.0;
    }
  }

  /**
   * checks whether there was already a distribution added to the history
   * 
   * @return		true if there's already a distribution
   */
  public boolean hasHistory() {
    return m_HasHistory;
  }

  /**
   * returns the index of the given instance, less than zero if not found
   * 
   * @param inst      the instance to look for
   * @return          the index of the instance
   */
  protected int find(Instance inst) {
    return Arrays.binarySearch(m_Instances, inst, m_Comparator);
  }

  /**
   * adds the given distribution to the history
   * 
   * @param inst      the instance to add the distribution for
   * @param dist      the distribution to add to the history
   */
  public void add(Instance inst, double[] dist) {
    int       index;
    int       i;

    index = find(inst);
    if (index < 0) {
      System.out.println("ERROR: cannot find instance in history!\n" + inst);
    }
    else {
      m_HasHistory = true;
      m_Count[index]++;
      for (i = 0; i < dist.length; i++) {
        m_Last[index][i]     = dist[i];
        m_Average[index][i] += dist[i];
      }
    }
  }

  /**
   * returns the count for the given instances, -1 if it cannot be found
   * @param inst      the instance to get the count for
   * @return          the count for the instance
   */
  public int getCount(Instance inst) {
    int       index;
    int       result;

    index = find(inst);
    if (index < 0)
      result = -1;
    else
      result = m_Count[index];

    return result;
  }

  /**
   * returns the last distribution for the given instance, only 0s if it cannot
   * be found
   * 
   * @param inst      the instance to retrieve the last distribution for
   * @return          the distribution for the given instance
   */
  public double[] getLast(Instance inst) {
    int       index;
    double[]  result;

    index  = find(inst);
    if (index < 0)
      result = new double[inst.numClasses()];
    else
      result = m_Last[index];

    return result;
  }

  /**
   * returns the average distribution for the given instance, only 0s if it
   * cannot be found
   * 
   * @param inst      the instance to retrieve the average distribution for
   * @return          the average distribution for the given instance
   */
  public double[] getAverage(Instance inst) {
    int       index;
    double[]  result;
    int       i;

    index  = find(inst);
    result = new double[inst.numClasses()];
    if (index > -1) {
      for (i = 0; i < result.length; i++)
        result[i] = m_Average[index][i] / m_Count[index];
    }
    
    return result;
  }

  /**
   * returns a string representation of the stored instances and statistics
   * 
   * @return		a string representation
   */
  @Override
  public String toString() {
    StringBuffer      result;
    int               i;
    int               n;
    double[]          dist;

    result = new StringBuffer();

    for (i = 0; i < m_Instances.length; i++) {
      result.append(m_Instances[i].toString());
      result.append("\n   " + "Count=" + m_Count[i]);
      result.append(", Last=");
      dist = m_Last[i];
      for (n = 0; n < dist.length; n++) {
        if (n > 0)
          result.append(",");
        result.append("" + dist[n]);
      }
      result.append(", Avg=");
      dist = getAverage(m_Instances[i]);
      for (n = 0; n < dist.length; n++) {
        if (n > 0)
          result.append(",");
        result.append("" + dist[n]);
      }
      result.append("\n");
    }

    return result.toString();
  }

  /**
   * for testing only, takes an ARFF filename as first argument
   * 
   * @param args	the commandline arguments
   * @throws Exception	if something goes wrong, e.g, file not found
   */
  public static void main(String[] args) throws Exception {
    FlipHistory     history;
    Instances       insts;
    double[]        dist;
    int             i;
    int             n;
    int             m;
    Random          rand;
    
    insts   = new Instances(new BufferedReader(new FileReader(args[0])));
    insts.setClassIndex(insts.numAttributes() - 1);
    history = new FlipHistory(insts);
    dist    = new double[insts.numClasses()];
    rand    = new Random();
    
    for (i = 0; i < insts.numInstances(); i++) {
      for (m = 0; m < 3; m++) {
        for (n = 0; n < dist.length; n++)
          dist[n] = rand.nextDouble();
        Utils.normalize(dist);
        history.add(insts.instance(i), dist);
      }
    }

    System.out.println(history);
  }
}
