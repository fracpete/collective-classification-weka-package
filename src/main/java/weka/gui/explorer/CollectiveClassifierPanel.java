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
 *    CollectiveClassifierPanel.java
 *    Copyright (C) 2013-2014 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.explorer;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Hashtable;
import java.util.Random;

import javax.swing.BorderFactory;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JViewport;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.CollectiveEvaluation;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.meta.YATSI;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.Loader;
import weka.gui.ConverterFileChooser;
import weka.gui.ExtensionFileFilter;
import weka.gui.GenericObjectEditor;
import weka.gui.Logger;
import weka.gui.PropertyPanel;
import weka.gui.ResultHistoryPanel;
import weka.gui.SaveBuffer;
import weka.gui.SysErrLog;
import weka.gui.TaskLogger;
import weka.gui.explorer.Explorer.CapabilitiesFilterChangeEvent;
import weka.gui.explorer.Explorer.CapabilitiesFilterChangeListener;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

/** 
 * This panel allows the user to select and configure a classifier, set the
 * attribute of the current dataset to be used as the class, and perform an
 * Experiment (like in the Experimenter) with this Classifier/Dataset
 * combination. The results of the experiment runs are stored in a result
 * history so that previous results are accessible. <p/>
 * 
 * Based on the ClassifierPanel code (by Len Trigg, Mark Hall and 
 * Richard Kirkby).
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2029 $
 */
public class CollectiveClassifierPanel 
  extends JPanel
  implements CapabilitiesFilterChangeListener, ExplorerPanel, LogHandler {
   
  /** for serialization. */
  private static final long serialVersionUID = 2078066653508312179L;

  /** the key for the model. */
  public final static String KEY_MODEL = "model";

  /** the key for the predictions. */
  public final static String KEY_PREDICTIONS = "predictions";

  /** the key for the errors. */
  public final static String KEY_ERRORS = "errors";
  
  /** the parent frame. */
  protected Explorer m_Explorer = null;

  /** Lets the user configure the classifier. */
  protected GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();

  /** The panel showing the current classifier selection. */
  protected PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor);
  
  /** The output area for classification results. */
  protected JTextArea m_OutText = new JTextArea(20, 40);

  /** The destination for log/status messages. */
  protected Logger m_Log = new SysErrLog();

  /** The buffer saving object for saving output. */
  protected SaveBuffer m_SaveOut = new SaveBuffer(m_Log, this);

  /** A panel controlling results viewing. */
  protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);
  
  /** the panel for the options (evalution, parameters, class). */
  protected JPanel m_PanelOptions = new JPanel(new BorderLayout());
  
  /** The type of evaluation: cross-validation/random split/test set. */
  protected JComboBox m_EvalCombo = new JComboBox(new String[]{"Cross-validation", "Random split", "Test set"});

  /** the label for the CV parameters. */
  protected JPanel m_CVPanel = new JPanel();
  
  /** The label for the number of folds. */
  protected JLabel m_CVFoldsLabel = new JLabel("Folds");

  /** the number of folds. */
  protected JTextField m_CVFoldsText = new JTextField("10", 10);

  /** The label for the CV seed value. */
  protected JLabel m_CVSeedLabel = new JLabel("Seed");

  /** the CV seed value. */
  protected JTextField m_CVSeedText = new JTextField("1", 10);

  /** The label for the CV swap folds. */
  protected JLabel m_CVSwapFoldsLabel = new JLabel("Swap folds");

  /** the CV swap folds checkbox. */
  protected JCheckBox m_CVSwapFoldsCheckBox = new JCheckBox();

  /** the label for the random split parameters. */
  protected JPanel m_SplitPanel = new JPanel();

  /** The label for the percentage for the random split. */
  protected JLabel m_SplitPercLabel = new JLabel("Percent");

  /** the percentage for the random split. */
  protected JTextField m_SplitPercText = new JTextField("10", 10);

  /** The label for the random split seed value. */
  protected JLabel m_SplitSeedLabel = new JLabel("Seed");

  /** the random split seed value. */
  protected JTextField m_SplitSeedText = new JTextField("1", 10);

  /** The label for the random split preserve order. */
  protected JLabel m_SplitPreserveOrderLabel = new JLabel("Preserve order");

  /** the random split preserve order checkbox. */
  protected JCheckBox m_SplitPreserveOrderCheckBox = new JCheckBox();

  /** the label for the test set parameters. */
  protected JPanel m_TestPanel = new JPanel();

  /** The label for the test set file. */
  protected JLabel m_TestFileLabel = new JLabel("Test set");

  /** the test set file button. */
  protected JButton m_TestFileButton = new JButton("...");

  /** Lets the user select the class column. */
  protected JComboBox m_ClassCombo = new JComboBox();
  
  /** Click to start running the experiment. */
  protected JButton m_StartBut = new JButton("Start");

  /** Click to stop a running experiment. */
  protected JButton m_StopBut = new JButton("Stop");

  /** Stop the class combo from taking up to much space. */
  private Dimension COMBO_SIZE = new Dimension(200, m_StartBut.getPreferredSize().height);

  /** The main set of instances we're playing with. */
  protected Instances m_Instances;

  /** The loader used to load the user-supplied test set (if any). */
  protected Loader m_TestLoader;
  
  /** A thread that classification runs in. */
  protected Thread m_RunThread;

  /** the file chooser for loading the test set. */
  protected ConverterFileChooser m_TestFileChooser;
  
  /** the current test set. */
  protected Instances m_TestSet;
  
  /** for saving models. */
  protected JFileChooser m_ModelFileChooser;
  
  /**
   * Creates the Experiment panel.
   */
  public CollectiveClassifierPanel() {
    m_TestFileChooser = new ConverterFileChooser();
    
    m_ModelFileChooser = new JFileChooser();
    ExtensionFileFilter filter = new ExtensionFileFilter("model", "Model files");
    m_ModelFileChooser.addChoosableFileFilter(filter);
    m_ModelFileChooser.setFileFilter(filter);
    
    m_OutText.setEditable(false);
    m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
    m_OutText.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    m_OutText.addMouseListener(new MouseAdapter() {
      @Override
      public void mouseClicked(MouseEvent e) {
	if ((e.getModifiers() & InputEvent.BUTTON1_MASK)
	    != InputEvent.BUTTON1_MASK) {
	  m_OutText.selectAll();
	}
      }
    });
    
    m_History.setBorder(BorderFactory.createTitledBorder("Result list (right-click for options)"));

    m_ClassifierEditor.setClassType(CollectiveClassifier.class);
    m_ClassifierEditor.setValue(new YATSI());
    m_ClassifierEditor.addPropertyChangeListener(new PropertyChangeListener() {
      public void propertyChange(PropertyChangeEvent e) {
	repaint();
      }
    });

    m_EvalCombo.setToolTipText("The type of evaluation to be performed");
    m_EvalCombo.setEnabled(false);
    m_EvalCombo.setPreferredSize(COMBO_SIZE);
    m_EvalCombo.setMaximumSize(COMBO_SIZE);
    m_EvalCombo.setMinimumSize(COMBO_SIZE);
    m_EvalCombo.setSelectedIndex(0);
    m_EvalCombo.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	int selected = m_EvalCombo.getSelectedIndex();
	if (selected == 0) {
	  m_PanelOptions.remove(m_SplitPanel);
	  m_PanelOptions.remove(m_TestPanel);
	  m_PanelOptions.add(m_CVPanel, BorderLayout.CENTER);
	}
	else if (selected == 1) {
	  m_PanelOptions.remove(m_CVPanel);
	  m_PanelOptions.remove(m_TestPanel);
	  m_PanelOptions.add(m_SplitPanel, BorderLayout.CENTER);
	}
	else if (selected == 2) {
	  m_PanelOptions.remove(m_CVPanel);
	  m_PanelOptions.remove(m_SplitPanel);
	  m_PanelOptions.add(m_TestPanel, BorderLayout.CENTER);
	}
	invalidate();
	validate();
	doLayout();
	repaint();
      }
    });

    m_CVFoldsText.setToolTipText("Number of folds for cross-validation");
    m_CVSeedText.setToolTipText("Seed value for randomizing data");
    m_CVSwapFoldsCheckBox.setToolTipText("Swaps train/test set");

    m_SplitPercText.setToolTipText("Percentage to use for training data");
    m_SplitSeedText.setToolTipText("Seed value for randomizing data");
    m_SplitPreserveOrderCheckBox.setToolTipText("Preserves the order in the data, suppresses randomization");

    m_TestFileButton.setToolTipText("Click to select test set");
    m_TestFileButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
	int retVal = m_TestFileChooser.showOpenDialog(CollectiveClassifierPanel.this);
	if (retVal != ConverterFileChooser.APPROVE_OPTION)
	  return;
	AbstractFileLoader loader = m_TestFileChooser.getLoader();
	try {
	  m_TestSet = loader.getDataSet();
	}
	catch (Exception ex) {
	  ex.printStackTrace();
	  JOptionPane.showMessageDialog(
	      CollectiveClassifierPanel.this, 
	      "Failed to load data from '" + m_TestFileChooser.getSelectedFile() + "':\n" + ex);
	  m_TestSet = null;
	}
      }
    });
    
    m_ClassCombo.setToolTipText("Select the attribute to use as the class");
    m_ClassCombo.setEnabled(false);
    m_ClassCombo.setPreferredSize(COMBO_SIZE);
    m_ClassCombo.setMaximumSize(COMBO_SIZE);
    m_ClassCombo.setMinimumSize(COMBO_SIZE);
    m_ClassCombo.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	updateCapabilitiesFilter(m_ClassifierEditor.getCapabilitiesFilter());
      }
    });

    m_StartBut.setToolTipText("Starts the evaluation");
    m_StartBut.setEnabled(false);
    m_StartBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	startClassifier();
      }
    });
    
    m_StopBut.setToolTipText("Stops a running evaluation");
    m_StopBut.setEnabled(false);
    m_StopBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	stopClassifier();
      }
    });
   
    m_History.setHandleRightClicks(false);
    // see if we can popup a menu for the selected result
    m_History.getList().addMouseListener(new MouseAdapter() {
      @Override
      public void mouseClicked(MouseEvent e) {
	if (((e.getModifiers() & InputEvent.BUTTON1_MASK)
	    != InputEvent.BUTTON1_MASK) || e.isAltDown()) {
	  int index = m_History.getList().locationToIndex(e.getPoint());
	  if (index != -1) {
	    String name = m_History.getNameAtIndex(index);
	    showPopup(name, e.getX(), e.getY());
	  } else {
	    showPopup(null, e.getX(), e.getY());
	  }
	}
      }
    });

    // Layout the GUI
    JPanel pClassifier = new JPanel();
    pClassifier.setBorder(
	BorderFactory.createCompoundBorder(
	    BorderFactory.createTitledBorder("Classifier"),
	    BorderFactory.createEmptyBorder(0, 5, 5, 5)));
    pClassifier.setLayout(new BorderLayout());
    pClassifier.add(m_CEPanel, BorderLayout.NORTH);
    
    m_PanelOptions.setBorder(
	BorderFactory.createCompoundBorder(
	    BorderFactory.createTitledBorder("Evaluation options"),
	    BorderFactory.createEmptyBorder(0, 5, 5, 5)));

    GridBagConstraints gbC;
    GridBagLayout gbL;

    // CV
    gbL = new GridBagLayout();
    m_CVPanel.setLayout(gbL);
    
    // CV/Folds
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 0;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVFoldsLabel, gbC);
    m_CVPanel.add(m_CVFoldsLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 0;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVFoldsText, gbC);
    m_CVPanel.add(m_CVFoldsText);
    
    // CV/Seed
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 1;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVSeedLabel, gbC);
    m_CVPanel.add(m_CVSeedLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 1;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVSeedText, gbC);
    m_CVPanel.add(m_CVSeedText);
    
    // CV/swap folds
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 2;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVSwapFoldsLabel, gbC);
    m_CVPanel.add(m_CVSwapFoldsLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 2;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_CVSwapFoldsCheckBox, gbC);
    m_CVPanel.add(m_CVSwapFoldsCheckBox);
    
    m_PanelOptions.add(m_CVPanel, BorderLayout.CENTER);
    
    // random split
    gbL = new GridBagLayout();
    m_SplitPanel.setLayout(gbL);
    
    // random split/percentage
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 0;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitPercLabel, gbC);
    m_SplitPanel.add(m_SplitPercLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 0;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitPercText, gbC);
    m_SplitPanel.add(m_SplitPercText);
    
    // random split/Seed
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 1;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitSeedLabel, gbC);
    m_SplitPanel.add(m_SplitSeedLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 1;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitSeedText, gbC);
    m_SplitPanel.add(m_SplitSeedText);
    
    // random split/preserve order
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 2;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitPreserveOrderLabel, gbC);
    m_SplitPanel.add(m_SplitPreserveOrderLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 2;
    gbC.gridx = 1;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_SplitPreserveOrderCheckBox, gbC);
    m_SplitPanel.add(m_SplitPreserveOrderCheckBox);
    
    // test set
    gbL = new GridBagLayout();
    m_TestPanel.setLayout(gbL);

    // test set/file
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 0;
    gbC.gridx = 0;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_TestFileLabel, gbC);
    m_TestPanel.add(m_TestFileLabel);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.NONE;
    gbC.gridy = 0;
    gbC.gridx = 1;
    gbC.weightx = 0;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    gbL.setConstraints(m_TestFileButton, gbC);
    m_TestPanel.add(m_TestFileButton);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 0;
    gbC.gridx = 2;
    gbC.weightx = 100;
    gbC.ipadx = 20;
    gbC.insets = new Insets(2, 5, 2, 5);
    JLabel label = new JLabel();
    gbL.setConstraints(label, gbC);
    m_TestPanel.add(label);
    
    // Evaluation
    JPanel pEval = new JPanel(new FlowLayout(FlowLayout.LEFT));
    pEval.add(new JLabel("Evaluation"));
    pEval.add(m_EvalCombo);
    
    m_PanelOptions.add(pEval, BorderLayout.NORTH);
    
    // class
    JPanel pClass = new JPanel();
    pClass.setLayout(new GridLayout(2, 2));
    pClass.add(m_ClassCombo);
    m_ClassCombo.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    JPanel ssButs = new JPanel();
    ssButs.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    ssButs.setLayout(new GridLayout(1, 2, 5, 5));
    ssButs.add(m_StartBut);
    ssButs.add(m_StopBut);

    pClass.add(ssButs);
    
    JPanel pOptionsButtons = new JPanel(new BorderLayout());
    pOptionsButtons.add(m_PanelOptions, BorderLayout.CENTER);
    pOptionsButtons.add(pClass, BorderLayout.SOUTH);
    
    JPanel pOutput = new JPanel();
    pOutput.setBorder(BorderFactory.createTitledBorder("Evaluation output"));
    pOutput.setLayout(new BorderLayout());
    final JScrollPane js = new JScrollPane(m_OutText);
    pOutput.add(js, BorderLayout.CENTER);
    js.getViewport().addChangeListener(new ChangeListener() {
      private int lastHeight;
      public void stateChanged(ChangeEvent e) {
	JViewport vp = (JViewport)e.getSource();
	int h = vp.getViewSize().height; 
	if (h != lastHeight) { // i.e. an addition not just a user scrolling
	  lastHeight = h;
	  int x = h - vp.getExtentSize().height;
	  vp.setViewPosition(new Point(0, x));
	}
      }
    });

    JPanel pOptionsHistory = new JPanel(new BorderLayout());
    pOptionsHistory.add(pOptionsButtons, BorderLayout.NORTH);
    pOptionsHistory.add(m_History, BorderLayout.CENTER);

    JPanel pOptionsHistoryOutput = new JPanel(new BorderLayout());
    pOptionsHistoryOutput.add(pOptionsHistory, BorderLayout.WEST);
    pOptionsHistoryOutput.add(pOutput, BorderLayout.CENTER);
    
    setLayout(new BorderLayout());
    add(pClassifier, BorderLayout.NORTH);
    add(pOptionsHistoryOutput, BorderLayout.CENTER);
  }

  /**
   * Sets the Logger to receive informational messages.
   *
   * @param newLog 	the Logger that will now get info messages
   */
  public void setLog(Logger newLog) {
    m_Log = newLog;
  }

  /**
   * Tells the panel to use a new set of instances.
   *
   * @param inst 	a set of Instances
   */
  public void setInstances(Instances inst) {
    m_Instances = inst;

    String[] attribNames = new String [m_Instances.numAttributes()];
    for (int i = 0; i < attribNames.length; i++) {
      String type = "";
      switch (m_Instances.attribute(i).type()) {
      case Attribute.NOMINAL:
	type = "(Nom) ";
	break;
      case Attribute.NUMERIC:
	type = "(Num) ";
	break;
      case Attribute.STRING:
	type = "(Str) ";
	break;
      case Attribute.DATE:
	type = "(Dat) ";
	break;
      case Attribute.RELATIONAL:
	type = "(Rel) ";
	break;
      default:
	type = "(???) ";
      }
      attribNames[i] = type + m_Instances.attribute(i).name();
    }
    m_ClassCombo.setModel(new DefaultComboBoxModel(attribNames));
    if (attribNames.length > 0) {
      if (inst.classIndex() == -1)
	m_ClassCombo.setSelectedIndex(attribNames.length - 1);
      else
	m_ClassCombo.setSelectedIndex(inst.classIndex());
      m_EvalCombo.setEnabled(true);
      m_ClassCombo.setEnabled(true);
      m_CVPanel.setEnabled(true);
      m_SplitPanel.setEnabled(true);
      m_TestPanel.setEnabled(true);
      m_StartBut.setEnabled(m_RunThread == null);
      m_StopBut.setEnabled(m_RunThread != null);
    }
    else {
      m_StartBut.setEnabled(false);
      m_StopBut.setEnabled(false);
    }
  }

  /**
   * Handles constructing a popup menu with visualization options.
   * 
   * @param name 	the name of the result history list entry clicked on by
   * 			the user
   * @param x 		the x coordinate for popping up the menu
   * @param y 		the y coordinate for popping up the menu
   */
  protected void showPopup(String name, int x, int y) {
    JPopupMenu 				result;
    JMenuItem 				menuitem;
    final String 			selectedName;
    final Hashtable<String,Object> 	additional;

    result = new JPopupMenu();
    selectedName = name;
    if (selectedName != null)
      additional = (Hashtable<String,Object>) m_History.getSelectedObject();
    else
      additional = new Hashtable<String,Object>();
    
    menuitem = new JMenuItem("View in main window");
    menuitem.setEnabled(selectedName != null);
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	m_History.setSingle(selectedName);
      }
    });
    result.add(menuitem);

    menuitem = new JMenuItem("View in separate window");
    menuitem.setEnabled(selectedName != null);
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	m_History.openFrame(selectedName);
      }
    });
    result.add(menuitem);

    menuitem = new JMenuItem("Save result buffer");
    menuitem.setEnabled(selectedName != null);
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	saveBuffer(selectedName);
      }
    });
    result.add(menuitem);

    menuitem = new JMenuItem("Delete result buffer");
    menuitem.setEnabled(selectedName != null);
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	m_History.removeResult(selectedName);
      }
    });
    result.add(menuitem);

    result.addSeparator();
    
    menuitem = new JMenuItem("Save model");
    menuitem.setEnabled((selectedName != null) && additional.containsKey(KEY_MODEL));
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	saveModel((Object[]) additional.get(KEY_MODEL));
      }
    });
    result.add(menuitem);

    result.addSeparator();
    
    menuitem = new JMenuItem("Visualize errors");
    menuitem.setEnabled((selectedName != null) && additional.containsKey(KEY_ERRORS));
    menuitem.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	visualizeClassifierErrors((VisualizePanel) additional.get(KEY_ERRORS));
      }
    });
    result.add(menuitem);

    result.show(m_History.getList(), x, y);
  }

  /**
   * Starts running the currently configured classifier.
   */
  protected void startClassifier() {
    if (m_RunThread == null) {
      synchronized (this) {
	m_StartBut.setEnabled(false);
	m_StopBut.setEnabled(true);
      }
      
      m_RunThread = new Thread() {
	@Override
	public void run() {
	  // set up everything:
	  m_Log.statusMessage("Setting up...");

	  try {
	    CollectiveEvaluation eval = null;
	    CollectiveClassifier classifier = (CollectiveClassifier) AbstractClassifier.makeCopy((CollectiveClassifier) m_ClassifierEditor.getValue());
	    String title = "";
	    boolean model = false;
		
	    m_Log.logMessage("Started evaluation for " + m_ClassifierEditor.getValue().getClass().getName());
	    if (m_Log instanceof TaskLogger)
	      ((TaskLogger)m_Log).taskStarted();
	    
	    // evaluating
	    m_Log.statusMessage("Evaluating...");

	    // cross-validation
	    if (m_EvalCombo.getSelectedIndex() == 0) {
	      title = "Cross-validation";
	      Instances train = new Instances(m_Instances);
	      train.setClassIndex(m_ClassCombo.getSelectedIndex());
	      eval = new CollectiveEvaluation(train);
	      int folds = Integer.parseInt(m_CVFoldsText.getText());
	      int seed  = Integer.parseInt(m_CVSeedText.getText());
	      eval.setSwapFolds(m_CVSwapFoldsCheckBox.isSelected());
	      eval.crossValidateModel(classifier, train, folds, new Random(seed));
	    }
	    // random split
	    else if (m_EvalCombo.getSelectedIndex() == 1) {
	      title = "Random split";
	      model = true;
	      Instances train = new Instances(m_Instances);
	      Instances test;
	      double percentage = Double.parseDouble(m_SplitPercText.getText());
	      int seed = Integer.parseInt(m_SplitSeedText.getText());
	      if (!m_SplitPreserveOrderCheckBox.isSelected())
		train.randomize(new Random(seed));
	      int trainSize = (int) Math.round(train.numInstances() * percentage / 100);
	      int testSize  = train.numInstances() - trainSize;
	      test  = new Instances(train, trainSize, testSize);
	      train = new Instances(train, 0, trainSize);
	      train.setClassIndex(m_ClassCombo.getSelectedIndex());
	      test.setClassIndex(m_ClassCombo.getSelectedIndex());
	      eval  = new CollectiveEvaluation(train);
	      if (classifier instanceof CollectiveClassifier)
		((CollectiveClassifier) classifier).buildClassifier(train, test);
	      else
		classifier.buildClassifier(train);
	      eval.evaluateModel(classifier, test);
	    }
	    // test set
	    else if (m_EvalCombo.getSelectedIndex() == 2) {
	      title = "Supplied test set";
	      model = true;
	      if (m_TestSet == null)
		throw new IllegalStateException("No test set set!");
	      Instances train = new Instances(m_Instances);
	      train.setClassIndex(m_ClassCombo.getSelectedIndex());
	      Instances test = new Instances(m_TestSet);
	      test.setClassIndex(m_ClassCombo.getSelectedIndex());
	      if (!train.equalHeaders(test))
		throw new IllegalStateException(train.equalHeadersMsg(test));
	      eval = new CollectiveEvaluation(train);
	      if (classifier instanceof CollectiveClassifier)
		((CollectiveClassifier) classifier).buildClassifier(train, test);
	      else
		classifier.buildClassifier(train);
	      eval.evaluateModel(classifier, test);
	    }
	    else {
	      throw new IllegalArgumentException("Unknown evaluation type: " + m_EvalCombo.getSelectedItem());
	    }

	    // assemble output
	    StringBuffer outBuff = new StringBuffer();
	    if (model) {
	      outBuff.append("=== Model ===\n");
	      outBuff.append("\n");
	      outBuff.append(classifier.toString());
	      outBuff.append("\n");
	      outBuff.append("\n");
	    }
	    outBuff.append(eval.toSummaryString("=== " + title + " ===\n", false));
	    
	    // additional information
	    Hashtable<String,Object> additional = new Hashtable<String,Object>();
	    // 1. model
	    if (model)
	      additional.put(KEY_MODEL, new Object[]{classifier, new Instances(m_Instances, 0)});
	    // 2. predictions
	    additional.put(KEY_PREDICTIONS, eval.predictions());
	    // 3. errors
	    DataGenerator generator = new DataGenerator(eval);
	    PlotData2D plotdata = generator.getPlotData();
	    plotdata.setPlotName(generator.getPlotInstances().relationName());
	    VisualizePanel visualizePanel = new VisualizePanel();
	    visualizePanel.addPlot(plotdata);
	    visualizePanel.setColourIndex(plotdata.getPlotInstances().classIndex());
	    if ((visualizePanel.getXIndex() == 0) && (visualizePanel.getYIndex() == 1)) {
	      try {
		visualizePanel.setXIndex(visualizePanel.getInstances().classIndex());  // class
		visualizePanel.setYIndex(visualizePanel.getInstances().classIndex() - 1);  // predicted class
	      }
	      catch (Exception e) {
		// ignored
	      }
	    }
	    additional.put(KEY_ERRORS, visualizePanel);
	    
	    String name = m_ClassifierEditor.getValue().getClass().getName().replaceAll("weka\\.classifiers\\.", "");
	    SimpleDateFormat df = new SimpleDateFormat("HH:mm:ss");
	    name = df.format(new Date()) + " - " + name;
	    m_History.addResult(name, outBuff);
	    m_History.addObject(name, additional);
	    m_History.setSingle(name);
	    m_Log.statusMessage("Evaluation finished.");
	    m_Log.statusMessage("OK");
	  }
	  catch (Exception ex) {
	    ex.printStackTrace();
	    m_Log.logMessage(ex.getMessage());
	    JOptionPane.showMessageDialog(
		CollectiveClassifierPanel.this,
		"Problem evaluating:\n" + ex.getMessage(),
		"Evaluation",
		JOptionPane.ERROR_MESSAGE);
	    m_Log.statusMessage("Problem evaluating");
	  }
	  finally {
	    synchronized (this) {
	      m_StartBut.setEnabled(true);
	      m_StopBut.setEnabled(false);
	      m_RunThread = null;
	    }
	    
	    if (m_Log instanceof TaskLogger)
              ((TaskLogger)m_Log).taskFinished();
	  }
	}
      };
      m_RunThread.setPriority(Thread.MIN_PRIORITY);
      m_RunThread.start();
    }
  }

  /**
   * Save the currently selected experiment output to a file.
   * 
   * @param name 	the name of the buffer to save
   */
  protected void saveBuffer(String name) {
    StringBuffer sb = m_History.getNamedBuffer(name);
    if (sb != null) {
      if (m_SaveOut.save(sb))
	m_Log.logMessage("Save successful.");
    }
  }

  /**
   * Save the currently selected experiment output to a file.
   * 
   * @param name 	the name of the buffer to save
   */
  protected void saveModel(Object[] data) {
    int retVal = m_ModelFileChooser.showSaveDialog(CollectiveClassifierPanel.this);
    if (retVal != JFileChooser.APPROVE_OPTION)
      return;
    try {
      SerializationHelper.writeAll(m_ModelFileChooser.getSelectedFile().getAbsolutePath(), data);
      m_Log.logMessage("Model saved successfully");
    }
    catch (Exception ex) {
      String msg = "Failed to save model to '" + m_ModelFileChooser.getSelectedFile() + "': " + ex;
      m_Log.logMessage(msg);
      JOptionPane.showMessageDialog(CollectiveClassifierPanel.this, msg);
    }
  }

  /**
   * Pops up a VisualizePanel for visualizing the data and errors for the
   * classifier from the currently selected item in the results list.
   * 
   * @param sp the VisualizePanel to pop up.
   */
  protected void visualizeClassifierErrors(VisualizePanel sp) {
    if (sp != null) {
      JFrame jf = new javax.swing.JFrame("Classifier Visualize: " + sp.getName());
      jf.setSize(600, 400);
      jf.getContentPane().setLayout(new BorderLayout());
      jf.getContentPane().add(sp, BorderLayout.CENTER);
      jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
      jf.setVisible(true);
    }
  }

  /**
   * Stops the currently running evaluation (if any).
   */
  protected void stopClassifier() {
    if (m_RunThread != null) {
      m_RunThread.interrupt();
      
      // This is deprecated (and theoretically the interrupt should do).
      m_RunThread.stop();
    }
  }
  
  /**
   * updates the capabilities filter of the GOE.
   * 
   * @param filter	the new filter to use
   */
  protected void updateCapabilitiesFilter(Capabilities filter) {
    Instances 		tempInst;
    Capabilities 	filterClass;

    if (filter == null) {
      m_ClassifierEditor.setCapabilitiesFilter(new Capabilities(null));
      return;
    }
    
    if (!ExplorerDefaults.getInitGenericObjectEditorFilter())
      tempInst = new Instances(m_Instances, 0);
    else
      tempInst = new Instances(m_Instances);
    tempInst.setClassIndex(m_ClassCombo.getSelectedIndex());

    try {
      filterClass = Capabilities.forInstances(tempInst);
    }
    catch (Exception e) {
      filterClass = new Capabilities(null);
    }
    
    // set new filter
    m_ClassifierEditor.setCapabilitiesFilter(filterClass);
  }
  
  /**
   * method gets called in case of a change event.
   * 
   * @param e		the associated change event
   */
  public void capabilitiesFilterChanged(CapabilitiesFilterChangeEvent e) {
    if (e.getFilter() == null)
      updateCapabilitiesFilter(null);
    else
      updateCapabilitiesFilter((Capabilities) e.getFilter().clone());
  }

  /**
   * Sets the Explorer to use as parent frame (used for sending notifications
   * about changes in the data).
   * 
   * @param parent	the parent frame
   */
  public void setExplorer(Explorer parent) {
    m_Explorer = parent;
  }
  
  /**
   * returns the parent Explorer frame.
   * 
   * @return		the parent
   */
  public Explorer getExplorer() {
    return m_Explorer;
  }
  
  /**
   * Returns the title for the tab in the Explorer.
   * 
   * @return 		the title of this tab
   */
  public String getTabTitle() {
    return "Collective";
  }
  
  /**
   * Returns the tooltip for the tab in the Explorer.
   * 
   * @return 		the tooltip of this tab
   */
  public String getTabTitleToolTip() {
    return "Collective classification";
  }
}
