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
 * RankedInstanceComparator.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.Comparator;

import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

/**
 * A comparator for the RankedInstance class. It can be used with or without
 * the class label. Missing values are sorted at the beginning.
 *
 * @see     Instance
 * @see     InstanceComparator
 *
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class RankedInstanceComparator 
  implements Comparator, Serializable {
  
  /** for serialization */
  private static final long serialVersionUID = -3175831412964203175L;
  
  /** the acutal comparator */
  protected InstanceComparator m_Comparator = null;
    
  /**
   * initializes the comparator and includes the class in the comparison 
   */
  public RankedInstanceComparator() {
    this(true);
  }
  
  /**
   * initializes the comparator  
   * 
   * @param includeClass	whether to include the class attribute
   */
  public RankedInstanceComparator(boolean includeClass) {
    super();
    
    m_Comparator = new InstanceComparator(includeClass);
  }
  
  /**
   * sets whether the class should be included (= TRUE) in the comparison
   * 
   * @param includeClass        whether to include the class in the comparison 
   */
  public void setIncludeClass(boolean includeClass) {
    m_Comparator.setIncludeClass(includeClass);
  }
  
  /**
   * returns TRUE if the class is included in the comparison
   * 
   * @return		true if the class attribute is included in the
   * 			comparison
   */
  public boolean getIncludeClass() {
    return m_Comparator.getIncludeClass();
  }

  /**
   * compares the two instances, returns -1 if o1 is smaller than o2, 0
   * if equal and +1 if greater. The method assumes that both instance objects
   * have the same attributes, they don't have to belong to the same dataset.
   * 
   * @param o1        the first instance to compare
   * @param o2        the second instance to compare
   * @return          returns -1 if o1 is smaller than o2, 0 if equal and +1 
   *                  if greater
   */
  public int compare(Object o1, Object o2) {
    int       result;
    
    if (    (o1 instanceof RankedInstance) 
         && (o2 instanceof RankedInstance) ) {
      result = m_Comparator.compare( ((RankedInstance) o1).getInstance(),
                                     ((RankedInstance) o2).getInstance() );
    }
    else {
      System.out.println("Tried to compare '" + o1 + "' with '" + o2 + "'!");
      result = 0;
    }

    return result;
  }
  
  /**
   * For testing only. Takes an ARFF-filename as first argument to perform
   * some tests. 
   * 
   * @param args	the commandline arguments
   * @throws Exception	if something goes wrong, e.g., file not found
   */
  public static void main(String[] args) throws Exception {
    Instances       inst;
    Comparator      comp;
    RankedInstance  i0;
    RankedInstance  i1;
    
    if (args.length == 0)
      return;
    
    // read instances
    inst = new Instances(new BufferedReader(new FileReader(args[0])));
    inst.setClassIndex(inst.numAttributes() - 1);
    
    i0 = new RankedInstance(inst.instance(0));
    i1 = new RankedInstance(inst.instance(1));
    
    // compare incl. class
    comp = new RankedInstanceComparator();
    System.out.println("\nIncluding the class");
    System.out.println("comparing 1. instance with 1.: " + comp.compare(i0, i0));
    System.out.println("comparing 1. instance with 2.: " + comp.compare(i0, i1));
    System.out.println("comparing 2. instance with 1.: " + comp.compare(i1, i0));
        
    // compare excl. class
    comp = new RankedInstanceComparator(false);
    System.out.println("\nExcluding the class");
    System.out.println("comparing 1. instance with 1.: " + comp.compare(i0, i0));
    System.out.println("comparing 1. instance with 2.: " + comp.compare(i0, i1));
    System.out.println("comparing 2. instance with 1.: " + comp.compare(i1, i0));
  }
}
