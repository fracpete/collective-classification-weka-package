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
 * CollectiveHelper.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;
  
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Contains various useful methods, e.g., for writing to a file. With the
 * main method one can generate a MD5 hash over a given String.
 *
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public class CollectiveHelper {
  /**
   * writes the content to the given file - used only for debugging
   * 
   * @param filename        the file to write to (is not appended!)
   * @param content         the content to write to the file
   */
  public static void writeToFile(String filename, String content) {
    writeToFile(filename, content, false);
  }
  
  /**
   * writes the content to the given file - used only for debugging
   * 
   * @param filename        the file to write to
   * @param content         the content to write to the file
   * @param append          whether the content is appended to the file
   */
  public static void writeToFile(String filename, String content, boolean append) {
    BufferedWriter        writer;
    
    try {
      writer = new BufferedWriter(new FileWriter(filename, append));
      writer.write(content);
      writer.newLine();
      writer.flush();
      writer.close();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * returns the temporary directory
   *
   * @return the path of the temporary directory
   */
  public static String getTempPath() {
    String        result;

    result = System.getProperty("java.io.tmpdir");
    if (!result.endsWith(System.getProperty("file.separator")))
      result += System.getProperty("file.separator");

    return result;
  }

  /**
   * writes the content to the given file (adds automatically the temp. path) 
   * - used only for debugging
   * 
   * @param filename        the file to write to (is not appended!)
   * @param content         the content to write to the file
   */
  public static void writeToTempFile(String filename, String content) {
    writeToFile(getTempPath() + filename, content);
  }
  
  /**
   * writes the content to the given file (temp. path is added automatically) 
   * - used only for debugging
   * 
   * @param filename        the file to write to
   * @param content         the content to write to the file
   * @param append          whether the content is appended to the file
   */
  public static void writeToTempFile( String filename, String content, 
                                      boolean append ) {
    writeToFile(getTempPath() + filename, content, append);
  }
  
  /**
   * Generates an MD5 hash over the given buffer. Returns NULL if something
   * goes wrong
   * @param buffer          the buffer to generate the MD5 for
   * @return                the generated MD5
   */
  public static byte[] generateMD5(byte[] buffer) {
    byte[]          result;
    MessageDigest   md5;

    result = null;

    try {
      md5    = MessageDigest.getInstance("MD5");
      result = md5.digest(buffer);
    }
    catch (NoSuchAlgorithmException e) {
      result = null;
    }

    return result;
  }

  /**
   * Generates an MD5 hash over the given String. Returns NULL if something
   * goes wrong
   * @param buffer          the String to generate the MD5 for
   * @return                the generated MD5
   */
  public static String generateMD5(String buffer) {
    byte[]      md5;
    String      result;
    String      hex;
    int         i;

    md5 = generateMD5(buffer.getBytes());
    if (md5 == null) {
      result = null;
    }
    else {
      result = "";
      for (i = 0; i < md5.length; i++) {
        hex = Integer.toHexString(md5[i]);
        while (hex.length() < 2)
          hex = "0" + hex;
        result += hex;
      }
    }

    return result;
  }

  /**
   * takes a string as parameter and generates a MD5 hash from it.
   * 
   * @param args	the commandline arguments
   */
  public static void main(String[] args) {
    if (args.length > 0) {
      for (int i = 0; i < args.length; i++)
        System.out.println(
            args[i] + "\n --> " + CollectiveHelper.generateMD5(args[i]));
    }
    else {
      System.out.println(
            CollectiveHelper.class.getName() 
          + " outputs a MD5 hash for a supplied string.");
    }
  }
}
