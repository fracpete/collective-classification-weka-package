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
 * Neighbors.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.lazy.ibk;

import java.io.Serializable;
import java.util.Arrays;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.KDTree;

/**
 * This class associates all the "k" neighbors of an instance with the
 * instance itself. It calculates the class for this instance (majority vote)
 * and the rank. In case of ties the class label of the nearest instance
 * is taken. As distance the euclidean distance is used.
 *
 * @see #getClassValue()
 * @see #getRank()
 * @version $Revision: 2019 $
 */
public class Neighbors
  implements Comparable, Serializable {

  /** for serialization */
  private static final long serialVersionUID = -2621385645290966136L;

  /** whether to use the naive search, searching a list, instead of KDTree */
  public boolean m_UseNaiveSearch;

  /** whether to output some information during building */
  private boolean m_Verbose;

  /** the KDTree to determine the neighbors - is only rebuilt if the
   * test/training sets differ */
  private static KDTree m_Tree = null;

  /** the training set reference to work with */
  private static Instances m_Trainset = null;

  /** the test set reference to work with */
  private static Instances m_Testset = null;

  /** the number of neighbors to use */
  private int m_KNN;

  /** the instance (after replace missing values) these neighbors are
   * associated with */
  private Instance m_Instance;

  /** the original instance these neighbors are associated with */
  private Instance m_InstanceOriginal;

  /** the current rank */
  private int m_Rank;

  /** the current class label */
  private double m_ClassValue;

  /** the list of neighbors */
  private Neighbor[] m_NeighborList;

  /** whether the class value was updated */
  private boolean m_Updated;

  /** indicates whether the neighborhood doesn't contain any missing classes anymore */
  private boolean m_Complete;

  /**
   * initializes the search, but the search still has to be initiated with
   * <code>find()</code>. The extra call of the <code>find()</code> is only
   * that other options can be set before the search is performed.
   * @param inst        the instance to find the neighbors for
   * @param kNN         the number of neighbors to look for
   * @param train       the training instances
   * @param test        the test instances
   * @see               #find()
   */
  public Neighbors(Instance inst, int kNN, Instances train, Instances test) {
    this(inst, inst, kNN, train, test);
  }

  /**
   * initializes the search, but the search still has to be initiated with
   * <code>find()</code>. The extra call of the <code>find()</code> is only
   * that other options can be set before the search is performed.
   * @param inst        the instance to find the neighbors for
   * @param orig        the original instance before applying the
   *                    ReplaceMissingValues filter
   * @param kNN         the number of neighbors to look for
   * @param train       the training instances
   * @param test        the test instances
   * @see               #find()
   */
  public Neighbors(Instance inst, Instance orig, int kNN, Instances train, Instances test) {
    m_KNN              = kNN;
    m_Instance         = (Instance) inst.copy();
    m_InstanceOriginal = (Instance) orig.copy();
    m_Verbose          = false;
    m_UseNaiveSearch   = false;
    m_Updated          = false;
    m_Complete         = false;
    m_Rank             = -1;

    // only rebuild tree if necessary!
    if ( (m_Trainset == null) || (m_Trainset != train) ) {
      m_Trainset = train;
      m_Testset  = test;
      m_Tree = null;
    }
  }

  /**
   * Sets whether to output some additional information
   *
   * @param verbose		if true more information will be printed
   */
  public void setVerbose(boolean verbose) {
    m_Verbose = verbose;
  }

  /**
   * Returns whether additional information is printed
   *
   * @return			true if more information is printed
   */
  public boolean getVerbose() {
    return m_Verbose;
  }

  /**
   * Sets whether to use the naive list search (= TRUE) or the KDTree (= FALSE)
   * is used for finding the neighbors
   *
   * @param useNaiveSearch	if true the naive list search is used instead of
   * 				the KDTree
   */
  public void setUseNaiveSearch(boolean useNaiveSearch) {
    m_UseNaiveSearch = useNaiveSearch;
  }

  /**
   * Returns whether the naive search or the KDTree is used to search for the
   * neighbors
   *
   * @return			true if the naive list search is used
   */
  public boolean getUseNaiveSearch() {
    return m_UseNaiveSearch;
  }

  /**
   * Invalidates the class and the rank. But the class only if the class label
   * is not already set via <code>updateClassValue()</code>
   *
   * @see     #updateClassValue()
   */
  public void invalidate() {
    m_Updated = false;
    m_Rank    = -1;
    if (getInstance().classIsMissing())
      m_ClassValue = Utils.missingValue();
  }

  /**
   * searches for the neighbors
   */
  public void find() {
    invalidate();

    if (getUseNaiveSearch())
      findNeighborsNaive();
    else
      findNeighborsKDTree();
  }

  /**
   * performs a naive search over all instances
   */
  protected void findNeighborsNaive() {
    Neighbor[]          allNeighbors;
    int                 i;
    int                 n;
    Instances           inst;
    int                 trainLength;
    int                 testLength;
    EuclideanDistance   dist;
    InstanceComparator  comp;
    int                 skipCount;

    m_NeighborList = new Neighbor[getKNN()];
    dist           = new EuclideanDistance(m_Trainset);
    allNeighbors   = null;

    try {
      trainLength  = m_Trainset.numInstances();
      testLength   = m_Testset.numInstances();
      allNeighbors = new Neighbor[trainLength + testLength - 1];
      comp         = new InstanceComparator();

      // training set
      n         = 0;
      skipCount = 0;
      inst      = m_Trainset;
      for (i = 0; i < inst.numInstances(); i++) {
        if (comp.compare(inst.instance(i), getInstance(false)) == 0) {
           skipCount++;
           // we can only skip one, otherwise we may get too few instances!
           // we'll inform the user later
           if (skipCount == 1)
             continue;
        }

        allNeighbors[n] = new Neighbor(inst.instance(i), dist.distance(getInstance(false), inst.instance(i)));
        n++;
      }

      // test set
      inst = m_Testset;
      for (i = 0; i < inst.numInstances(); i++) {
        if (comp.compare(inst.instance(i), getInstance(false)) == 0) {
           skipCount++;
           // we can only skip one, otherwise we may get too few instances!
           // we'll inform the user later
           if (skipCount == 1)
             continue;
        }

        allNeighbors[n] = new Neighbor(inst.instance(i), dist.distance(getInstance(false), inst.instance(i)));
        n++;
      }

      // would we have skipped more than 1?
      if ( (getVerbose()) && (skipCount > 1) )
        System.out.println("WARNING: " + getInstance(false) + ": would have skipped " + skipCount + "!");

      // sort neighbors according to distance and get the k nearest
      Arrays.sort(allNeighbors);
      for (i = 0; i < getKNN(); i++)
        m_NeighborList[i] = new Neighbor(allNeighbors[i].getInstance(), allNeighbors[i].getDistance());
      allNeighbors = null;
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * uses a KDTree to locate the neighbors for the given instance
   */
  protected void findNeighborsKDTree() {
    Instances           allInstances;
    Instances           neighbors;
    double[]            distances;
    int                 i;
    int                 n;
    int                 skipCount;
    InstanceComparator  comp;

    try {
      // build tree and find k+1 neighbors (the instance we're looking for is
      // also in that list, hence "+1"! we skip it later)
      if (m_Tree == null) {
        if (getVerbose())
          System.out.println("Rebuilding KDTree...");
        m_Tree = new KDTree();
        allInstances = new Instances(m_Trainset);
        for (i = 0; i < m_Testset.numInstances(); i++)
          allInstances.add(m_Testset.instance(i));
        m_Tree.setInstances(allInstances);
      }
      neighbors = m_Tree.kNearestNeighbours(getInstance(false), getKNN() + 1);
      distances = m_Tree.getDistances();

      // create neighbor list
      m_NeighborList = new Neighbor[getKNN()];
      comp           = new InstanceComparator();
      n              = 0;
      skipCount      = 0;
      for (i = 0; i < neighbors.numInstances(); i++) {
        // is it the instance we want the neighbors for? -> skip it
        if (comp.compare(getInstance(false), neighbors.instance(i)) == 0) {
          skipCount++;
          // we can only skip one, otherwise we get a NULL below!
          // we'll inform the user later
          if (skipCount == 1)
            continue;
        }

        m_NeighborList[n] = new Neighbor(neighbors.instance(i), distances[i]);
        if (n == getKNN() - 1)
          break;
        n++;
      }

      // would we have skipped more than 1?
      if ( (getVerbose()) && (skipCount > 1) )
        System.out.println("WARNING: " + getInstance(false) + ": would have skipped " + skipCount + "!");
    }
    catch (Exception e) {
      e.printStackTrace();
      System.out.println("Using naive search to determine neighbors...");
      findNeighborsNaive();
    }
  }

  /**
   * returns the number of neighbors
   *
   * @return		the number of neighbors
   */
  public int getKNN() {
    return m_KNN;
  }

  /**
   * returns the instance the neighbors are determined for (returns the
   * original instance, not having the missing values replaced)
   *
   * @return		the instance for which the neighbors are determined
   */
  public Instance getInstance() {
    return getInstance(true);
  }

  /**
   * returns the instance the neighbors are determined for, either the original
   * one or the the one after replacing missing values
   *
   * @param original	if true, the original instance is returned
   * @return		either the original or the transformed instance
   */
  public Instance getInstance(boolean original) {
    if (original)
      return m_InstanceOriginal;
    else
      return m_Instance;
  }

  /**
   * calculates the rank and the class label
   */
  protected void calculate() {
    int[]       count;
    int         i;
    int         maxFirst;
    int         maxSecond;

    // count occurrences for each label
    count = new int[getInstance().numClasses()];
    for (i = 0; i < getKNN(); i++) {
      if (m_NeighborList[i] == null)
        continue;
      if (!m_NeighborList[i].getInstance().classIsMissing())
        count[(int) m_NeighborList[i].getInstance().classValue()]++;
    }

    // get two most common labels
    maxFirst     = 0;
    maxSecond    = 0;
    m_ClassValue = 1;   // if there's no known class in the neighborhood, we pick the first
    for (i = 0; i < count.length; i++) {
      if (count[i] > maxFirst) {
        maxSecond = maxFirst;
        maxFirst  = count[i];
        // the class value is the one with the highest rank
        m_ClassValue = i;
        continue;
      }
      if (count[i] > maxSecond) {
        maxSecond = count[i];
      }
    }

    // rank is difference between two most common classes
    m_Rank = maxFirst - maxSecond;

    // do we have a tie? -> take the nearest class label
    if (m_Rank == 0) {
      for (i = 0; i < m_NeighborList.length; i++) {
        if (!m_NeighborList[i].getInstance().classIsMissing()) {
          m_ClassValue = m_NeighborList[i].getInstance().classValue();
          m_Rank       = 1;
          break;
        }
      }
    }
  }

  /**
   * Returns the rank of this neighbor, i.e., the difference between the
   * number of instances with known class labels for the two class labels.
   * E.g., 10 neighbors, 5 of class 1, 3 of class 2 and 2 unknown ones
   * returns (5-3) = 2 as rank.
   *
   * @return			the rank if the neighbor
   */
  public int getRank() {
    // was rank already calculated?
    if (!isComplete())
      calculate();

    return m_Rank;
  }

  /**
   * Returns the current class associated with the neighbors, i.e., majority
   * vote, in case of a tie, the first class label.
   *
   * @return			the current class label
   */
  public double getClassValue() {
    double          result;

    if (!getInstance().classIsMissing()) {
      result = getInstance().classValue();
    }
    else {
      if (Utils.isMissingValue(m_ClassValue))
        calculate();
      result = m_ClassValue;
    }

    return result;
  }

  /**
   * returns TRUE if the neighborhood has no more unlabeled neighbors
   *
   * @return		true if all neighbors are labeled
   */
  public boolean isComplete() {
    int       i;
    boolean   result;

    result = true;

    if (!m_Complete) {
      for (i = 0; i < m_NeighborList.length; i++) {
        if (m_NeighborList[i].getInstance().classIsMissing()) {
          result = false;
          break;
        }
      }
      m_Complete = result;
    }

    return result;
  }

  /**
   * whether the class value was updated after the last invalidate() call
   *
   * @see 		#updateClassValue()
   * @see 		#invalidate()
   * @return		true if the class label was updated
   */
  public boolean isUpdated() {
    return m_Updated;
  }

  /**
   * sets the class value of the instance to the one calculated, but only
   * if the class value was missing
   */
  public void updateClassValue() {
    if (getInstance().classIsMissing()) {
      m_Updated = true;
      getInstance().setClassValue(getClassValue());
    }
  }

  /**
   * Compares this object with the specified object for order.
   * If this object has a higher rank than the given one, it returns -1.
   *
   * @param o		the object to compare with
   * @return		the comparison result
   */
  public int compareTo(Object o) {
    int          result;
    Neighbors    ns;

    ns = (Neighbors) o;

    if (getRank() > ns.getRank())
      result = -1;
    else if (getRank() < ns.getRank())
      result = 1;
    else
      result = 0;

    return result;
  }

  /**
   * returns the neighborlist in a string representation
   *
   * @return		a string representation of this object
   */
  @Override
  public String toString() {
    String        result;
    int           i;

    result = "rank=" + getRank() + ", complete=" + isComplete() + ", updated=" + isUpdated() + ", inst=" + getInstance(false) + ", orig=" + getInstance(true);
    for (i = 0; i < m_NeighborList.length; i++)
      result += "\n   " + m_NeighborList[i];

    return result;
  }


  /* ********************* other classes ************************** */


  /**
   * This class is just for sortinf instances according to their distance.
   */
  protected class Neighbor
    implements Comparable, Serializable {

    /** for serialization */
    private static final long serialVersionUID = 5923706472440530618L;

    /** the distance */
    protected double m_Distance;

    /** the instance the distance is associated with */
    protected Instance m_Instance;

    /**
     * initializes the Neighbor
     *
     * @param inst	the instance
     * @param dist	the distance
     */
    public Neighbor(Instance inst, double dist) {
      m_Instance = inst;
      m_Distance = dist;
    }

    /**
     * returns the distance associated with the instance
     *
     * @return		the associated distance
     */
    public double getDistance() {
      return m_Distance;
    }

    /**
     * returns the instance
     *
     * @return		the instance
     */
    public Instance getInstance() {
      return m_Instance;
    }

    /**
     * Compares this object with the specified object for order.
     * If this object has a smaller distance than the given one, it returns -1.
     *
     * @param o		the object to compare with
     * @return		the comparison result
     */
    public int compareTo(Object o) {
      int       result;
      Neighbor  nb;

      nb = (Neighbor) o;

      if (getDistance() < nb.getDistance())
        result = -1;
      else if (getDistance() > nb.getDistance())
        result = 1;
      else
        result = 0;

      return result;
    }

    /**
     * Indicates whether some other object is "equal to" this one. But only
     * in regard to the distance!
     *
     * @param o		the object to compare with
     * @return		true if both objects are equal
     */
    @Override
    public boolean equals(Object o) {
      return (compareTo(o) == 0);
    }

    /**
     * returns the neighbor in a string representation
     *
     * @return		a string representation
     */
    @Override
    public String toString() {
      return "dist=" + getDistance() + ", inst=" + getInstance();
    }
  }
}

