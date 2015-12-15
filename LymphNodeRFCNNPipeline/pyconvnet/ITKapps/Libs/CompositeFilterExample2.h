/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//  Software Guide : BeginLatex
//
//  The composite filter we will build combines three filters: a gradient
//  magnitude operator, which will calculate the first-order derivative of
//  the image; a thresholding step to select edges over a given strength;
//  and finally a rescaling filter, to ensure the resulting image data is
//  visible by scaling the intensity to the full spectrum of the output
//  image type.
//
//  Since this filter takes an image and produces another image (of
//  identical type), we will specialize the ImageToImageFilter:
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
//  Software Guide : EndCodeSnippet
//  Software Guide : BeginLatex
//
//  Next we include headers for the component filters:
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
#include "itkGradientMagnitudeImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
//  Software Guide : EndCodeSnippet
#include "itkNumericTraits.h"
//  Software Guide : BeginLatex
//
//  Now we can declare the filter itself.  It is within the ITK namespace,
//  and we decide to make it use the same image type for both input and
//  output, thus the template declaration needs only one parameter.
//  Deriving from \code{ImageToImageFilter} provides default behavior for
//  several important aspects, notably allocating the output image (and
//  making it the same dimensions as the input).
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
namespace itk {
template <class TInputImageType, class TOutputImageType>
class CompositeExampleImageFilter2 :
    public ImageToImageFilter<TInputImageType, TOutputImageType>
{
public:
//  Software Guide : EndCodeSnippet
//  Software Guide : BeginLatex
//
//  Next we have the standard declarations, used for object creation with
//  the object factory:
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
  typedef CompositeExampleImageFilter2               Self;
  typedef ImageToImageFilter<TInputImageType,TInputImageType> Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;
//  Software Guide : EndCodeSnippet
  itkNewMacro(Self);
  itkTypeMacro(CompositeExampleImageFilter2, ImageToImageFilter);
  void PrintSelf( std::ostream& os, Indent indent ) const;
//  Software Guide : BeginLatex
//
//  Here we declare an alias (to save typing) for the image's pixel type,
//  which determines the type of the threshold value.  We then use the
//  convenience macros to define the Get and Set methods for this parameter.
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
  typedef typename TInputImageType::PixelType PixelType;
  itkGetMacro( Threshold, PixelType);
  itkSetMacro( Threshold, PixelType);
//  Software Guide : EndCodeSnippet
protected:
  CompositeExampleImageFilter2();
//  Software Guide : BeginLatex
//
//  Now we can declare the component filter types, templated over the
//  enclosing image type:
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
protected:
  typedef ThresholdImageFilter< TInputImageType >                     ThresholdType;
  typedef GradientMagnitudeImageFilter< TInputImageType, TInputImageType > GradientType;
  typedef RescaleIntensityImageFilter< TInputImageType, TOutputImageType >  RescalerType;
//  Software Guide : EndCodeSnippet
    void GenerateData();
private:
  CompositeExampleImageFilter2(Self&);   // intentionally not implemented
  void operator=(const Self&);          // intentionally not implemented
//  Software Guide : BeginLatex
//
//  The component filters are declared as data members, all using the smart
//  pointer types.
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
  typename GradientType::Pointer     m_GradientFilter;
  typename ThresholdType::Pointer    m_ThresholdFilter;
  typename RescalerType::Pointer     m_RescaleFilter;
  PixelType m_Threshold;
};
} /* namespace itk */
//  Software Guide : EndCodeSnippet
//  Software Guide : BeginLatex
//
//  The constructor sets up the pipeline, which involves creating the
//  stages, connecting them together, and setting default parameters.
//
//  Software Guide : EndLatex
namespace itk
{
//  Software Guide : BeginCodeSnippet
template <class TInputImageType, class TOutputImageType>
CompositeExampleImageFilter2<TInputImageType,TOutputImageType>
::CompositeExampleImageFilter2()
{
  m_Threshold = 1;
  m_GradientFilter = GradientType::New();
  m_ThresholdFilter = ThresholdType::New();
  m_ThresholdFilter->SetInput( m_GradientFilter->GetOutput() );
  m_RescaleFilter = RescalerType::New();
  m_RescaleFilter->SetInput( m_ThresholdFilter->GetOutput() );
  m_RescaleFilter->SetOutputMinimum(
                                  NumericTraits<PixelType>::NonpositiveMin());
  m_RescaleFilter->SetOutputMaximum(NumericTraits<PixelType>::max());
}
//  Software Guide : EndCodeSnippet
//  Software Guide : BeginLatex
//
//  The \code{GenerateData()} is where the composite magic happens.  First,
//  we connect the first component filter to the inputs of the composite
//  filter (the actual input, supplied by the upstream stage).  Then we
//  graft the output of the last stage onto the output of the composite,
//  which ensures the filter regions are updated.  We force the composite
//  pipeline to be processed by calling \code{Update()} on the final stage,
//  then graft the output back onto the output of the enclosing filter, so
//  it has the result available to the downstream filter.
//
//  Software Guide : EndLatex
//  Software Guide : BeginCodeSnippet
template <class TInputImageType, class TOutputImageType>
void
CompositeExampleImageFilter2<TInputImageType,TOutputImageType>::
GenerateData()
{
  m_GradientFilter->SetInput( this->GetInput() );
  m_ThresholdFilter->ThresholdBelow( this->m_Threshold );
  m_RescaleFilter->GraftOutput( this->GetOutput() );
  m_RescaleFilter->Update();
  this->GraftOutput( m_RescaleFilter->GetOutput() );
}
//  Software Guide : EndCodeSnippet
//  Software Guide : BeginLatex
//
//  Finally we define the \code{PrintSelf} method, which (by convention)
//  prints the filter parameters.  Note how it invokes the superclass to
//  print itself first, and also how the indentation prefixes each line.
//
//  Software Guide : EndLatex
//
//  Software Guide : BeginCodeSnippet
template <class TInputImageType, class TOutputImageType>
void
CompositeExampleImageFilter2<TInputImageType, TOutputImageType>::
PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);
  os
    << indent << "Threshold:" << this->m_Threshold
    << std::endl;
}
} /* end namespace itk */

