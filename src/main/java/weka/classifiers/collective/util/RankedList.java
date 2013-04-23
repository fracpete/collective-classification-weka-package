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
 * RankedList.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

/**
 * This class resembles a simple sorted array of Instance's with their
 * according distributions. 
 *
 *
 * @author      FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class RankedList
  implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 2439017952170416353L;

  /** the sorted array of instances with their distributions */
  private RankedInstance[] m_List = null;

  /** for sorting and finding */
  private Comparator m_Comparator = null;

  /**
   * initializes the list with the given instances
   *
   * @param inst      the dataset to create the sorted list from
   */
  public RankedList(Instances inst) {
    super();
    
    m_List = new RankedInstance[inst.numInstances()];
    for (int i = 0; i < m_List.length; i++)
      m_List[i] = new RankedInstance(inst.instance(i));

    m_Comparator = new RankedInstanceComparator(false);
    
    // sort list
    Arrays.sort(m_List, m_Comparator);
  }

  /**
   * initializes the list with the given instances and initializes
   * the disitributions with the given distribution
   *
   * @param inst      the dataset to create the sorted list from
   * @param dist      the initial distribution
   */
  public RankedList(Instances inst, double[] dist) {
    this(inst);

    try {
      for (int i = 0; i < m_List.length; i++)
        addDistribution(inst.instance(i), dist);
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * tries to return the associated instance, throws an Exception if it cannot
   * find it
   *
   * @param inst        the instance to find
   * @return		the ranked instance for the given instance
   * @throws Exception  if the instance cannot be found
   */
  public RankedInstance find(Instance inst) throws Exception {
    int       index;

    index = Arrays.binarySearch(m_List, new RankedInstance(inst), m_Comparator);
    if (index < 0)
      throw new Exception(  "Cannot find '" + inst + "' at index '" 
                          + StrictMath.abs(index) + "' -> " 
                          + m_List[StrictMath.abs(index)].getInstance() );
    
    return m_List[index];
  }

  /**
   * adds the distribution for the 
   *
   * @param inst        the instance to add the distribution for
   * @param dist        the distribution to add
   * @throws Exception  if the instance cannot be found in the list
   */
  public void addDistribution(Instance inst, double[] dist) throws Exception {
    find(inst).addDistribution(dist);
  }

  /**
   * returns the distribution for the instance
   *
   * @param inst        the instance to return the distribution for
   * @return		the distribution for the instance
   * @throws Exception  if the instance cannot be found in the list
   */
  public double[] getDistribution(Instance inst) throws Exception {
    return find(inst).getDistribution();
  }

  /**
   * returns the stored list in a string representation
   *
   * @return            the complete list as string
   */
  @Override
  public String toString() {
    StringBuffer        buf;
    int                 i;

    buf = new StringBuffer();
    for (i = 0; i < m_List.length; i++)
      buf.append(m_List[i].toString() + "\n");

    return buf.toString();
  }

  /**
   * For testing only. Takes as first argument an ARFF file.
   * the second argument is an optional position of an instance to retrieve
   * (testing the retrieving), if not given a random instance is taken.
   * 
   * @param args	the commandline arguments
   * @throws Exception	if something goes wrong, e.g., file not found
   */
  public static void main(String[] args) throws Exception {
    RankedList        list;
    int               index;
    Instances         inst;
    
    if (args.length > 0) {
      // load arff file
      inst = new Instances(new BufferedReader(new FileReader(args[0])));
      list = new RankedList(inst);

      // determine index
      if (args.length > 1)
        index = Integer.parseInt(args[1]);
      else
        index = new Random().nextInt(inst.numInstances());

      // print distribution
      System.out.println("ARFF file: " + args[0]);
      System.out.println("index    : " + index);
      System.out.println("wanted   : " + inst.instance(index));
      System.out.println("found    : " + list.find(inst.instance(index)).getInstance());
    }
  }
}
